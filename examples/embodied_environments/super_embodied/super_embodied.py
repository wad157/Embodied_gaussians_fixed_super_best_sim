# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import warp
from scipy.spatial.transform import Rotation

from embodied_gaussians import (
    Body,
    EmbodiedGaussiansBuilder,
    EmbodiedGaussiansEnvironment,
    Ground,
    SoftBody,
    read_ground,
)

from .psm_lnd_kinematics import PSMLNDKinematics

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parents[2]

SCENE_BODIES_PATH = (
    repo_root / "data/super/grasp5_native/bodies_v9_dense_0p5mm_rigid_tissue"
)
GROUND_PATH = SCENE_BODIES_PATH / "ground_plane.json"
TISSUE_PATH = SCENE_BODIES_PATH / "tissue.json"
GROUND_BODY_PATH = SCENE_BODIES_PATH / "ground.json"
SCENE_METADATA_PATH = SCENE_BODIES_PATH / "build_metadata.json"
ADAPTIVE_TISSUE_PATH = (
    repo_root
    / "data/super/grasp5_native/tissue_multiview_v1/"
    "soft_tissue_adaptive_v1/tissue_soft_adaptive.npz"
)
PAPER_PBD_TISSUE_PATH = (
    repo_root
    / "data/super/grasp5_native/tissue_multiview_v1/"
    "paper_pbd_tissue_v11_uniform_surface_centroid_ellipsoids/"
    "tissue_paper_pbd_centroid_gaussians.npz"
)
PAPER_CONSTRAINT_TISSUE_PATH = (
    repo_root
    / "data/super/grasp5_native/tissue_multiview_v1/"
    "paper_pbd_tissue_v15_denser_mild_paper_constraints_centroid_ellipsoids/"
    "tissue_paper_pbd_centroid_gaussians.npz"
)

PSM_URDF_PATH = repo_root / "data/super/psm_robot/psm.urdf"
PSM_MIMIC_MAP_PATH = repo_root / "data/super/psm_robot/psm_mimic_map.json"
PSM_SURFACE_GAUSSIANS_PATH = (
    repo_root / "data/super/psm_robot/psm_surface_gaussians.npz"
)
PSM_LND_POSE_DRIVER_PATH = repo_root / "data/super/psm_robot/psm_lnd_pose_driver.npz"
PSM_PAPER_POSE_DRIVER_PATH = (
    repo_root / "data/super/psm_tracking/psm_paper_exact_pose_driver.npz"
)
PSM_PAPER_ROBUST_POSE_DRIVER_PATH = (
    repo_root / "data/super/psm_tracking/psm_paper_pose_driver.npz"
)
PSM_HYBRID_POSE_DRIVER_PATH = (
    repo_root / "data/super/psm_tracking/psm_hybrid_pose_driver.npz"
)
PSM_PART_CORRECTED_POSE_DRIVER_PATH = (
    repo_root / "data/super/psm_tracking/psm_part_corrected_pose_driver.npz"
)
PSM_DEPTH_THEN_VISUAL_POSE_DRIVER_PATH = (
    repo_root
    / "data/super/psm_tracking/psm_depth_then_visual_pose_driver_candidate.npz"
)
PSM_REGISTERED_LND_POSE_DRIVER_PATH = (
    repo_root / "data/super/psm_tracking/psm_registered_lnd_pose_driver.npz"
)
PSM_RAW_KINEMATICS_ROOT = (
    repo_root / "data/super/psm_raw_kinematics_v1（纯机器人学版本）"
)
PSM_RAW_GUI_ROOT = PSM_RAW_KINEMATICS_ROOT / "gui_v1"
PSM_RAW_URDF_PATH = PSM_RAW_GUI_ROOT / "psm_raw.urdf"
PSM_RAW_MIMIC_MAP_PATH = PSM_RAW_GUI_ROOT / "psm_raw_mimic_map.json"
PSM_RAW_SURFACE_GAUSSIANS_PATH = (
    PSM_RAW_GUI_ROOT / "psm_raw_surface_gaussians.npz"
)
PSM_RAW_POSE_DRIVER_PATH = (
    PSM_RAW_GUI_ROOT / "psm_raw_gui_pose_driver.npz"
)
PSM_RAW_LND_MODEL_PATH = PSM_RAW_KINEMATICS_ROOT / "model.json"
PSM_RAW_REGISTRATION_REPORT_PATH = (
    PSM_RAW_GUI_ROOT / "registration_report.json"
)
PSM_RAW_P420006_GUI_ROOT = PSM_RAW_KINEMATICS_ROOT / "gui_p420006_v1"
PSM_RAW_P420006_URDF_PATH = (
    PSM_RAW_P420006_GUI_ROOT / "psm_p420006.urdf"
)
PSM_RAW_P420006_MIMIC_MAP_PATH = (
    PSM_RAW_P420006_GUI_ROOT / "psm_p420006_mimic_map.json"
)
PSM_RAW_P420006_SURFACE_GAUSSIANS_PATH = (
    PSM_RAW_P420006_GUI_ROOT / "psm_p420006_surface_gaussians.npz"
)
PSM_RAW_P420006_POSE_DRIVER_PATH = (
    PSM_RAW_P420006_GUI_ROOT / "psm_p420006_gui_pose_driver.npz"
)
PSM_RAW_P420006_REGISTRATION_REPORT_PATH = (
    PSM_RAW_P420006_GUI_ROOT / "registration_report.json"
)
PSM_RAW_P420006_STEREO_VISUAL_ROOT = (
    repo_root
    / "data/super/psm_visual_calibration/"
    "raw_p420006_stereo_v1/gui_v1"
)
PSM_RAW_P420006_STEREO_VISUAL_POSE_DRIVER_PATH = (
    PSM_RAW_P420006_STEREO_VISUAL_ROOT
    / "psm_p420006_stereo_visual_gui_pose_driver.npz"
)
PSM_RAW_P420006_SAM2_ONLINE_ROOT = (
    repo_root
    / "data/super/psm_visual_calibration/raw_p420006_stereo_v1/"
    "surgicalsam2_stereo_sequence_v1/online_stereo_cma_v2/gui_v1"
)
PSM_RAW_P420006_SAM2_ONLINE_POSE_DRIVER_PATH = (
    PSM_RAW_P420006_SAM2_ONLINE_ROOT
    / "psm_p420006_sam2_online_gui_pose_driver.npz"
)
PSM_RAW_PAPER_LND_SAM2_ONLINE_ROOT = (
    repo_root
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_v1"
    / "surgicalsam2_stereo_sequence_v1/online_stereo_cma_v1/gui_v1"
)
PSM_RAW_PAPER_LND_URDF_PATH = (
    PSM_RAW_PAPER_LND_SAM2_ONLINE_ROOT / "psm_paper_lnd.urdf"
)
PSM_RAW_PAPER_LND_MIMIC_MAP_PATH = (
    PSM_RAW_PAPER_LND_SAM2_ONLINE_ROOT
    / "psm_paper_lnd_mimic_map.json"
)
PSM_RAW_PAPER_LND_SURFACE_GAUSSIANS_PATH = (
    PSM_RAW_PAPER_LND_SAM2_ONLINE_ROOT
    / "psm_paper_lnd_surface_gaussians.npz"
)
PSM_RAW_PAPER_LND_SAM2_ONLINE_POSE_DRIVER_PATH = (
    PSM_RAW_PAPER_LND_SAM2_ONLINE_ROOT
    / "psm_paper_lnd_gui_pose_driver.npz"
)
PSM_RAW_PAPER_LND_REGISTRATION_REPORT_PATH = (
    PSM_RAW_PAPER_LND_SAM2_ONLINE_ROOT / "registration_report.json"
)
PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_ROOT = (
    repo_root
    / "data/super/psm_visual_calibration/"
    "raw_paper_lnd_first_stereo_static_v1/gui_v1"
)
PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_URDF_PATH = (
    PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_ROOT / "psm_paper_lnd.urdf"
)
PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_MIMIC_MAP_PATH = (
    PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_ROOT
    / "psm_paper_lnd_mimic_map.json"
)
PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_SURFACE_GAUSSIANS_PATH = (
    PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_ROOT
    / "psm_paper_lnd_surface_gaussians.npz"
)
PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_POSE_DRIVER_PATH = (
    PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_ROOT
    / "psm_paper_lnd_gui_pose_driver.npz"
)
PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_REGISTRATION_REPORT_PATH = (
    PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_ROOT / "registration_report.json"
)
PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_ROOT = (
    repo_root
    / "data/super/psm_visual_calibration/"
    "raw_paper_lnd_first_stereo_se3_fixed_q5_v1/gui_v1"
)
PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_URDF_PATH = (
    PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_ROOT / "psm_paper_lnd.urdf"
)
PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_MIMIC_MAP_PATH = (
    PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_ROOT
    / "psm_paper_lnd_mimic_map.json"
)
PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_SURFACE_GAUSSIANS_PATH = (
    PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_ROOT
    / "psm_paper_lnd_surface_gaussians.npz"
)
PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_POSE_DRIVER_PATH = (
    PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_ROOT
    / "psm_paper_lnd_gui_pose_driver.npz"
)
PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_REGISTRATION_REPORT_PATH = (
    PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_ROOT
    / "registration_report.json"
)
PSM_RAW_PAPER_LND_SAM2_MULTIANCHOR_CLOSEDJAW_ROOT = (
    repo_root
    / "data/super/psm_visual_calibration/"
    "raw_paper_lnd_stereo_multianchor_v2/"
    "surgicalsam2_multianchor_parts_v4（paper_LND+视觉矫正）/"
    "online_stereo_cma_closedjaw_v1/gui_v1"
)
PSM_RAW_PAPER_LND_MULTIANCHOR_CLOSEDJAW_URDF_PATH = (
    PSM_RAW_PAPER_LND_SAM2_MULTIANCHOR_CLOSEDJAW_ROOT
    / "psm_paper_lnd.urdf"
)
PSM_RAW_PAPER_LND_MULTIANCHOR_CLOSEDJAW_MIMIC_MAP_PATH = (
    PSM_RAW_PAPER_LND_SAM2_MULTIANCHOR_CLOSEDJAW_ROOT
    / "psm_paper_lnd_mimic_map.json"
)
PSM_RAW_PAPER_LND_MULTIANCHOR_CLOSEDJAW_SURFACE_GAUSSIANS_PATH = (
    PSM_RAW_PAPER_LND_SAM2_MULTIANCHOR_CLOSEDJAW_ROOT
    / "psm_paper_lnd_surface_gaussians.npz"
)
PSM_RAW_PAPER_LND_SAM2_MULTIANCHOR_CLOSEDJAW_POSE_DRIVER_PATH = (
    PSM_RAW_PAPER_LND_SAM2_MULTIANCHOR_CLOSEDJAW_ROOT
    / "psm_paper_lnd_gui_pose_driver.npz"
)
PSM_RAW_PAPER_LND_MULTIANCHOR_CLOSEDJAW_REGISTRATION_REPORT_PATH = (
    PSM_RAW_PAPER_LND_SAM2_MULTIANCHOR_CLOSEDJAW_ROOT
    / "registration_report.json"
)
PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_CLOSEDJAW_ROOT = (
    repo_root
    / "data/super/psm_visual_calibration/"
    "raw_paper_lnd_stereo_dense_contact_v3/"
    "surgicalsam2_multianchor_parts_dense_contact_v5/"
    "online_stereo_cma_closedjaw_dense_contact_v2/"
    "gui_dense_contact_v2"
)
PSM_RAW_PAPER_LND_DENSE_CONTACT_CLOSEDJAW_URDF_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_CLOSEDJAW_ROOT
    / "psm_paper_lnd.urdf"
)
PSM_RAW_PAPER_LND_DENSE_CONTACT_CLOSEDJAW_MIMIC_MAP_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_CLOSEDJAW_ROOT
    / "psm_paper_lnd_mimic_map.json"
)
PSM_RAW_PAPER_LND_DENSE_CONTACT_CLOSEDJAW_SURFACE_GAUSSIANS_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_CLOSEDJAW_ROOT
    / "psm_paper_lnd_surface_gaussians.npz"
)
PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_CLOSEDJAW_POSE_DRIVER_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_CLOSEDJAW_ROOT
    / "psm_paper_lnd_gui_pose_driver.npz"
)
PSM_RAW_PAPER_LND_DENSE_CONTACT_CLOSEDJAW_REGISTRATION_REPORT_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_CLOSEDJAW_ROOT
    / "registration_report.json"
)
PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_SE3_ONLY_ROOT = (
    repo_root
    / "data/super/psm_visual_calibration/"
    "raw_paper_lnd_stereo_dense_contact_v3/"
    "surgicalsam2_multianchor_parts_dense_contact_v5/"
    "online_stereo_cma_se3_only_dense_contact_v3/"
    "gui_se3_only_v1"
)
PSM_RAW_PAPER_LND_DENSE_CONTACT_SE3_ONLY_URDF_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_SE3_ONLY_ROOT
    / "psm_paper_lnd.urdf"
)
PSM_RAW_PAPER_LND_DENSE_CONTACT_SE3_ONLY_MIMIC_MAP_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_SE3_ONLY_ROOT
    / "psm_paper_lnd_mimic_map.json"
)
PSM_RAW_PAPER_LND_DENSE_CONTACT_SE3_ONLY_SURFACE_GAUSSIANS_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_SE3_ONLY_ROOT
    / "psm_paper_lnd_surface_gaussians.npz"
)
PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_SE3_ONLY_POSE_DRIVER_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_SE3_ONLY_ROOT
    / "psm_paper_lnd_gui_pose_driver.npz"
)
PSM_RAW_PAPER_LND_DENSE_CONTACT_SE3_ONLY_REGISTRATION_REPORT_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_SE3_ONLY_ROOT
    / "registration_report.json"
)
PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_ROOT = (
    repo_root
    / "data/super/psm_visual_calibration/"
    "raw_paper_lnd_stereo_dense_contact_v4/"
    "surgicalsam2_multianchor_parts_dense_contact_v6/"
    "online_stereo_cma_se3_unbounded_dense_contact_v4/"
    "gui_unbounded_xyz_v1"
)
PSM_RAW_PAPER_LND_DENSE_CONTACT_UNBOUNDED_XYZ_URDF_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_ROOT
    / "psm_paper_lnd.urdf"
)
PSM_RAW_PAPER_LND_DENSE_CONTACT_UNBOUNDED_XYZ_MIMIC_MAP_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_ROOT
    / "psm_paper_lnd_mimic_map.json"
)
PSM_RAW_PAPER_LND_DENSE_CONTACT_UNBOUNDED_XYZ_SURFACE_GAUSSIANS_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_ROOT
    / "psm_paper_lnd_surface_gaussians_dense_v2_first_stereo_rgb.npz"
)
PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_POSE_DRIVER_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_ROOT
    / "psm_paper_lnd_gui_pose_driver.npz"
)
PSM_RAW_PAPER_LND_DENSE_CONTACT_UNBOUNDED_XYZ_REGISTRATION_REPORT_PATH = (
    PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_ROOT
    / "registration_report.json"
)
PSM_POSE_DRIVER_PATHS = {
    "raw_kinematics": PSM_RAW_POSE_DRIVER_PATH,
    "raw_p420006": PSM_RAW_P420006_POSE_DRIVER_PATH,
    "raw_p420006_stereo_visual": (
        PSM_RAW_P420006_STEREO_VISUAL_POSE_DRIVER_PATH
    ),
    "raw_p420006_sam2_online": (
        PSM_RAW_P420006_SAM2_ONLINE_POSE_DRIVER_PATH
    ),
    "raw_paper_lnd_sam2_online": (
        PSM_RAW_PAPER_LND_SAM2_ONLINE_POSE_DRIVER_PATH
    ),
    "raw_paper_lnd_first_stereo_static_q5": (
        PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_POSE_DRIVER_PATH
    ),
    "raw_paper_lnd_first_stereo_se3_fixed_q5": (
        PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_POSE_DRIVER_PATH
    ),
    "raw_paper_lnd_sam2_multianchor_closedjaw": (
        PSM_RAW_PAPER_LND_SAM2_MULTIANCHOR_CLOSEDJAW_POSE_DRIVER_PATH
    ),
    "raw_paper_lnd_sam2_dense_contact_closedjaw": (
        PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_CLOSEDJAW_POSE_DRIVER_PATH
    ),
    "raw_paper_lnd_sam2_dense_contact_se3_only": (
        PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_SE3_ONLY_POSE_DRIVER_PATH
    ),
    "raw_paper_lnd_sam2_dense_contact_unbounded_xyz": (
        PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_POSE_DRIVER_PATH
    ),
    "strict": PSM_LND_POSE_DRIVER_PATH,
    "registered_lnd": PSM_REGISTERED_LND_POSE_DRIVER_PATH,
    "paper": PSM_PAPER_POSE_DRIVER_PATH,
    "paper_robust": PSM_PAPER_ROBUST_POSE_DRIVER_PATH,
    "hybrid": PSM_HYBRID_POSE_DRIVER_PATH,
    "corrected": PSM_PART_CORRECTED_POSE_DRIVER_PATH,
    "depth_then_visual": PSM_DEPTH_THEN_VISUAL_POSE_DRIVER_PATH,
}
PSM_LND_POSE_REPORT_PATH = (
    repo_root / "data/super/psm_robot/psm_lnd_pose_driver_report.json"
)
PSM_LND_MODEL_PATH = (
    repo_root / "data/super/grasp5_offline_demo/instruments/psm1_lnd_model.json"
)
TABLE_FRAME_PATH = repo_root / "data/super/table_frame.json"
SUPER_DATASET_PATH = repo_root / "data/super/grasp5_offline_demo"
ROBOTS_PATH = SUPER_DATASET_PATH / "robots.json"
PSM_BASE_LINK_NAME = "PSM1_psm_base_link"
PSM_TISSUE_CONTACT_LINK_NAMES = (
    "PSM1_tool_wrist_sca_ee_link_1",
    "PSM1_tool_wrist_sca_ee_link_2",
)
PSM_TISSUE_JAW_CONTACT_LINK_NAMES = PSM_TISSUE_CONTACT_LINK_NAMES
# Warp joint_q slot order for PSM URDF.  Must match what Warp produces after
# parsing the articulation tree.  Fixed joints do not get joint_q slots.
PSM_WARP_JOINT_Q_ORDER = [
    "yaw",
    "pitch_2",
    "pitch_1",
    "pitch_4",
    "pitch_3",
    "pitch_5",
    "pitch",
    "insertion",
    "roll",
    "wrist_pitch",
    "wrist_yaw",
    "jaw_mimic_2",
    "jaw_mimic_1",
    "jaw",
]

TISSUE_GAUSSIAN_SCALE = 1.0
GROUND_GAUSSIAN_SCALE = 1.0
TISSUE_PARTICLE_RADIUS_SCALE = 1.0
PAPER_TISSUE_MODES = frozenset({"paper_pbd", "paper_soft"})
SOFT_TISSUE_MODES = frozenset({*PAPER_TISSUE_MODES, "adaptive_soft"})

# Legacy Neo-Hookean values retained only by ``paper_soft`` for regression.
PAPER_SOFT_TISSUE_YOUNG_MODULUS_PA = 50.0
PAPER_SOFT_TISSUE_POISSON_RATIO = 0.35
# The reconstructed surface is already the observed, gravity-loaded
# equilibrium configuration.  Treating it as an unloaded zero-stress rest
# shape and applying gravity again double-loads it.  Keep gravity compensated
# until an unloaded rest shape/prestress is recovered after material
# calibration.
PAPER_SOFT_TISSUE_GRAVITY_M_S2 = 0.0
PAPER_SOFT_TISSUE_VELOCITY_DAMPING_PER_SECOND = 10.0
# Keep geometric material recovery in position space, but feed only a small
# fraction of that correction into the next substep's inertia.  This does not
# alter distance/volume/shape stiffness or the online stiffness optimizer; it
# only suppresses the long displacement tail produced by repeatedly carrying
# a local press velocity through twelve substeps.
PAPER_SOFT_TISSUE_MATERIAL_PROJECTION_VELOCITY_SCALE = 0.14
PAPER_SOFT_TISSUE_CONTACT_PROJECTION_VELOCITY_SCALE = 0.35
# Repair near-collapsed elements locally before their large gradients can feed
# contact chatter back into the jaw patch. This is deliberately far below the
# visual-residual mapper's 0.30 acceptance floor and does not globally freeze
# the mesh: only vertices incident on an unsafe tetrahedron are reverted.
PAPER_SOFT_TISSUE_MATERIAL_MIN_VOLUME_RATIO = 0.01
PAPER_SOFT_TISSUE_MATERIAL_ITERATIONS = 6
PAPER_SOFT_TISSUE_MATERIAL_RELAXATION = 0.10
# Fixed Liang-et-al.-style volumetric PBD parameters used by ``paper_pbd``.
# Moderately firm reset baseline for the five-node prescribed patch.  The old
# 0.20/0.004 field localized the pull into a visible bump, while the historical
# 0.35/0.006 field propagated too far. Use a point below that historical setting
# and additional projection passes to strengthen transfer without restoring
# the long-range rigid response. No regional material truth is exposed.
PAPER_CONSTRAINT_DISTANCE_STIFFNESS = 0.31
PAPER_CONSTRAINT_VOLUME_STIFFNESS = 1.0e10
PAPER_CONSTRAINT_SHAPE_STIFFNESS = 0.0058
# Eight passes converge the three local constraint families after imposing the
# true five-node kinematic boundary. Unlike the removed explicit support-node
# overwrite, every additional pass remains a physical PBD propagation step and
# leaves all neighboring nodes available to RGB residual/stiffness correction.
PAPER_CONSTRAINT_MATERIAL_ITERATIONS = 8
PAPER_CONSTRAINT_MATERIAL_RELAXATION = 1.0
# One swept contact solve per physical substep is cheaper and more robust than
# three repeated nearest-triangle queries every second substep.  The gentler
# 0.03 mm general cap remains above the measured worst jaw translation (about
# 0.023 mm/substep), preserving q7 tangential closure.  The oriented top-sheet
# normal uses a separate 0.008 mm cap below so approach indentation cannot
# consume that full budget or flatten the tissue reserved for later gripping.
PAPER_SOFT_TISSUE_SURFACE_MAX_CORRECTION_M = 0.00003
PAPER_SOFT_TISSUE_TOP_BARRIER_MAX_CORRECTION_M = 0.000008
PAPER_SOFT_TISSUE_SURFACE_ITERATIONS = 1
PAPER_SOFT_TISSUE_POST_CONTACT_MATERIAL_ITERATIONS = 0
PAPER_SOFT_TISSUE_FINAL_BARRIER_MAX_CORRECTION_M = 0.0
PAPER_SOFT_TISSUE_CONTACT_SUBSTEP_STRIDE = 1
PAPER_SOFT_TISSUE_CONTACT_SPREAD_LAYERS = 0
PAPER_SOFT_TISSUE_TOOL_SAMPLE_SPACING_M = 0.00065
# The distal 5 mm supplies q7 closure/friction and bilateral grasp candidates.
# Its front 3.0 mm may enter the reconstructed surface.  Keep a minimal 0.01 mm
# top-sheet normal-barrier band immediately behind it so the validated geometry
# remains tip < barrier <= jaw length.  The remaining rear 1.99 mm stays
# available for tangential contact/grip candidates but cannot prescribe a
# downward displacement into tissue that the jaws have not closed around yet.
PAPER_SOFT_TISSUE_JAW_CONTACT_DISTAL_LENGTH_M = 0.005
PAPER_SOFT_TISSUE_TOP_BARRIER_DISTAL_LENGTH_M = 0.00301
PAPER_SOFT_TISSUE_TOP_BARRIER_TIP_ALLOWANCE_M = 0.0030
PAPER_SOFT_TISSUE_TOP_SUPPORT_RADIUS_M = 0.001
PAPER_SOFT_TISSUE_TOP_SUPPORT_DEPTH_M = 0.008
# Direct jaw contact is the only source of indentation.  The former weak
# one-ring support still prescribed a downward displacement outside the hit
# triangle and obscured whether the XPBD solid itself transmitted pressure.
# Leave all neighbouring motion to distance+volume+shape projection.
PAPER_SOFT_TISSUE_TOP_SUPPORT_WEIGHT_SCALE = 0.0
# The explicit pressure-shoulder displacement is deliberately disabled. It
# was an external position source, not an XPBD constraint, so repeated contact
# substeps could inject volume and create a visibly inflated patch. The narrow
# direct jaw footprint below remains active; neighbouring vertices now move
# only through the distance/volume/shape material projection.
PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_RADIUS_M = 0.0
PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_DEPTH_M = 0.0
PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_UPWARD_SCALE = 0.0
PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_OUTWARD_SCALE = 0.0
PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_BIAS_DIRECTION_WORLD = (0.0, 0.0, 0.0)
PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_BIAS_START_M = 0.0
PAPER_SOFT_TISSUE_TOP_BARRIER_LATERAL_TOLERANCE_M = 0.00030
# Direct downward projection is restricted to a roughly 2.5 mm physical-node
# core around each actual top-sheet hit.  Once bilateral contact exists, the
# same radius clips the direct jaw patch around its measured center.  Natural
# XPBD response can remain outside this core, but no artificial contact delta
# is prescribed there.  This is a contact-locality setting, not stiffness.
PAPER_SOFT_TISSUE_TOP_BARRIER_CONTACT_PATCH_RADIUS_M = 0.0025
# The oriented sheet barrier activates only very near the reconstructed plane.
# Ordinary jaw triangle contact keeps the independent 0.4 mm contact margin,
# so reducing this clearance does not remove q7/grip candidates.
PAPER_SOFT_TISSUE_TOP_CLEARANCE_M = 0.00025
PAPER_SOFT_TISSUE_CONTACT_MIN_VOLUME_RATIO = 0.03
# A grasp requires three consecutive solves with contact samples on both jaws.
# Two healthy contacted nodes per jaw become explicit positional controls u_t
# in that jaw's local frame. The contact solver reports penetration including
# the 0.4 mm contact margin, so 2.8 mm here permits at most about 2.4 mm of
# actual signed overlap at capture.  This deliberately permissive setting can
# recover deeper bilateral contact, but may also latch a badly penetrated pose.
PAPER_SOFT_TISSUE_GRIP_MAX_CAPTURE_PENETRATION_M = 0.0028
PAPER_SOFT_TISSUE_GRIP_CLOSED_ANGLE_MAX_RAD = 0.13
PAPER_SOFT_TISSUE_GRIP_RELEASE_ANGLE_MIN_RAD = 0.15
PAPER_SOFT_TISSUE_GRIP_RELEASE_ANGLE_DELTA_RAD = 0.08
PAPER_SOFT_TISSUE_GRIP_WIDE_OPEN_ANGLE_RAD = 0.50
PAPER_SOFT_TISSUE_GRIP_ANGLE_MOTION_EPSILON_RAD = 1.0e-4
PAPER_SOFT_TISSUE_GRIP_MIN_CONTACT_SAMPLES_PER_JAW = 8
PAPER_SOFT_TISSUE_GRIP_NEAREST_SURFACE_PARTICLES = 4
PAPER_SOFT_TISSUE_GRIP_MAX_PATCH_SEPARATION_M = 0.006
PAPER_SOFT_TISSUE_GRIP_ACTIVATION_STEPS = 3
PAPER_SOFT_TISSUE_GRIP_MIN_CAPTURE_VOLUME_RATIO = 0.20
# Zero-history compliant-XPBD attachment.  The correction cap remains the
# final geometric guard; compliance prevents a moving kinematic frame from
# snapping the four light surface nodes ahead of their incident tetrahedra.
PAPER_SOFT_TISSUE_GRIP_COMPLIANCE_M_PER_N = 0.05
PAPER_SOFT_TISSUE_GRIP_RELAXATION = 1.0
PAPER_SOFT_TISSUE_GRIP_MAX_CORRECTION_M = 0.00010
PAPER_SOFT_TISSUE_GRIP_TRANSFER_LAYERS = 1
# Four jaw-frame u_t particles remain the only full-weight anchors. After
# capture, one 4.0 mm top-sheet ring receives Gaussian-decayed compliant
# targets from its nearest direct anchor, allowing a small outside patch to
# lift without turning the complete tissue into a rigid grasp.
PAPER_SOFT_TISSUE_GRIP_SUPPORT_RADIUS_M = 0.0040
PAPER_SOFT_TISSUE_GRIP_SUPPORT_GENERATIONS = 1

# The geometric triangle-skin contact parameters are shared by paper_soft and
# the retained adaptive_soft fallback.  They do not alter material stiffness.
ADAPTIVE_TISSUE_YOUNG_MODULUS_PA = 50.0
ADAPTIVE_TISSUE_POISSON_RATIO = 0.40
ADAPTIVE_TISSUE_GRAVITY_M_S2 = 0.0
ADAPTIVE_TISSUE_TOOL_SAMPLE_SPACING_M = 0.00050
ADAPTIVE_TISSUE_CONTACT_SPREAD_LAYERS = 0
ADAPTIVE_TISSUE_TOP_CONTACT_SUPPORT_RADIUS_M = 0.00125
ADAPTIVE_TISSUE_TOP_CONTACT_SUPPORT_DEPTH_M = 0.0035
ADAPTIVE_TISSUE_VELOCITY_DAMPING_PER_SECOND = 1.0
ADAPTIVE_TISSUE_CONTACT_MARGIN_M = 0.0004
ADAPTIVE_TISSUE_CONTACT_QUERY_DISTANCE_M = 0.010
ADAPTIVE_TISSUE_CONTACT_MAX_CORRECTION_M = 0.00005
ADAPTIVE_TISSUE_CONTACT_MIN_VOLUME_RATIO = 0.0
ADAPTIVE_TISSUE_MATERIAL_MIN_VOLUME_RATIO = 0.0
ADAPTIVE_TISSUE_ENABLE_KINEMATIC_GAP_GUARD = False
ADAPTIVE_TISSUE_KINEMATIC_MIN_GAP_M = ADAPTIVE_TISSUE_CONTACT_MARGIN_M
PAPER_SOFT_TISSUE_JAW_FRICTION_COEFFICIENT = 1.5

# The reconstructed adaptive tissue is already an observed equilibrium shape.
# Gravity compensation avoids inventing a large unmodelled prestress merely to
# keep that rest shape from sagging, so low-modulus contact remains usable.
SUPER_VISUAL_FORCE_LR_MEANS = 0.0002
SUPER_VISUAL_FORCE_LR_QUATS = 0.0001
# Keep one optimizer iteration below the soft-force clamps so the interaction
# count controls physical force instead of saturating immediately.
SUPER_VISUAL_FORCE_KP = 0.05
SUPER_VISUAL_FORCE_MAX_FORCE_N = 0.005
SUPER_VISUAL_FORCE_MAX_MOMENT_NM = 0.00005
SUPER_VISUAL_FORCE_ROBUST_LOSS_BETA = 0.05
SUPER_SOFT_VISUAL_FORCE_MAX_GAUSSIAN_N = 3.0e-5
SUPER_SOFT_VISUAL_FORCE_MAX_PARTICLE_N = 3.0e-5
SUPER_SOFT_VISUAL_FORCE_MAX_TOTAL_N = 0.02
# Keep the mass-aware clamp, but do not suppress RGB shape correction to an
# invisible level. This value passed the 300-step v9 deformation gate and is
# still five times below the rejected 5 m/s^2 flying-particle candidate.
SUPER_SOFT_VISUAL_FORCE_MAX_PARTICLE_ACCELERATION_M_S2 = 1.0
SUPER_SOFT_VISUAL_FORCE_SPREAD_LAYERS = 0

PSM_ARTICULATION_INDEX = 0


def compute_lnd_body_velocities(
    timestamps: np.ndarray,
    poses_xyz_xyzw: np.ndarray,
    body_com: np.ndarray,
) -> np.ndarray:
    """Build world-frame angular/COM velocities for timestamped pose replay."""
    if len(timestamps) < 2:
        return np.zeros((*poses_xyz_xyzw.shape[:2], 6), dtype=np.float32)
    rotations = Rotation.from_quat(poses_xyz_xyzw[..., 3:].reshape(-1, 4))
    rotation_matrices = rotations.as_matrix().reshape(
        *poses_xyz_xyzw.shape[:2], 3, 3
    )
    com_world = poses_xyz_xyzw[..., :3] + np.einsum(
        "flij,lj->fli", rotation_matrices, body_com
    )
    velocities = np.zeros((*poses_xyz_xyzw.shape[:2], 6), dtype=np.float64)
    dt = np.diff(timestamps)
    if np.any(dt <= 0.0):
        raise ValueError("LND timestamps must be strictly increasing")
    velocities[1:, :, 3:] = np.diff(com_world, axis=0) / dt[:, None, None]
    relative_rotation = rotation_matrices[1:] @ np.swapaxes(
        rotation_matrices[:-1], -1, -2
    )
    rotation_vectors = Rotation.from_matrix(relative_rotation.reshape(-1, 3, 3))
    velocities[1:, :, :3] = rotation_vectors.as_rotvec().reshape(
        len(dt), poses_xyz_xyzw.shape[1], 3
    ) / dt[:, None, None]
    velocities[0] = velocities[1]
    return velocities.astype(np.float32)


def load_body(path: Path) -> Body:
    with open(path, "r") as f:
        return Body.model_validate(json.load(f))


def urdf_actuated_joint_order(urdf_path: Path) -> list[str]:
    supported_paths = {
        PSM_URDF_PATH.resolve(),
        PSM_RAW_URDF_PATH.resolve(),
        PSM_RAW_P420006_URDF_PATH.resolve(),
        PSM_RAW_PAPER_LND_URDF_PATH.resolve(),
        PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_URDF_PATH.resolve(),
        PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_URDF_PATH.resolve(),
        PSM_RAW_PAPER_LND_MULTIANCHOR_CLOSEDJAW_URDF_PATH.resolve(),
        PSM_RAW_PAPER_LND_DENSE_CONTACT_CLOSEDJAW_URDF_PATH.resolve(),
        PSM_RAW_PAPER_LND_DENSE_CONTACT_SE3_ONLY_URDF_PATH.resolve(),
        PSM_RAW_PAPER_LND_DENSE_CONTACT_UNBOUNDED_XYZ_URDF_PATH.resolve(),
    }
    if urdf_path.resolve() not in supported_paths:
        raise ValueError(
            "SUPER PSM joint order is only defined for the legacy and raw "
            f"URDF assets, got {urdf_path}"
        )
    return list(PSM_WARP_JOINT_Q_ORDER)


def load_mimic_config(path: Path = PSM_MIMIC_MAP_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_table_from_camera(path: Path = TABLE_FRAME_PATH) -> np.ndarray:
    with open(path, "r") as f:
        table_frame = json.load(f)
    return np.asarray(table_frame["X_table_camera"], dtype=np.float32)


def expand_psm_q7_to_joint_dict(q7: list[float] | np.ndarray, mimic_cfg: dict) -> dict[str, float]:
    q7_arr = np.asarray(q7, dtype=np.float32)
    input_names = mimic_cfg["input_joint_names"]
    if len(q7_arr) != len(input_names):
        raise ValueError(f"Expected {len(input_names)} PSM input joints, got {len(q7_arr)}")
    q_by_joint: dict[str, float] = {}
    input_to_urdf = mimic_cfg["input_to_urdf_joint"]
    for input_name, value in zip(input_names, q7_arr):
        q_by_joint[input_to_urdf[input_name]] = float(value)
    for mimic_joint_name, spec in mimic_cfg["mimic"].items():
        source_name = spec["source"]
        q_by_joint[mimic_joint_name] = (
            q_by_joint[source_name] * float(spec.get("multiplier", 1.0))
            + float(spec.get("offset", 0.0))
        )
    return q_by_joint


def expand_psm_q7_to_urdf_order(
    q7: list[float] | np.ndarray,
    mimic_cfg: dict | None = None,
    joint_order: list[str] | None = None,
) -> np.ndarray:
    if mimic_cfg is None:
        mimic_cfg = load_mimic_config()
    if joint_order is None:
        joint_order = urdf_actuated_joint_order(PSM_URDF_PATH)
    q_by_joint = expand_psm_q7_to_joint_dict(q7, mimic_cfg)
    missing = [name for name in joint_order if name not in q_by_joint]
    if missing:
        raise KeyError(f"Missing PSM joint values for URDF joints: {missing}")
    return np.asarray([q_by_joint[name] for name in joint_order], dtype=np.float32)


def load_initial_psm_q_full(
    q7: list[float] | np.ndarray | None = None,
    mimic_cfg: dict | None = None,
    joint_order: list[str] | None = None,
) -> np.ndarray:
    if q7 is None:
        with open(ROBOTS_PATH, "r") as f:
            robots = json.load(f)
        # super_best robots.json uses key "PSM1"
        q7 = robots["PSM1"]["states"][0]["q"]
    return expand_psm_q7_to_urdf_order(
        q7,
        mimic_cfg=mimic_cfg,
        joint_order=joint_order,
    )


def add_psm_surface_gaussians(
    builder: EmbodiedGaussiansBuilder,
    robot_body_count: int,
    path: Path = PSM_SURFACE_GAUSSIANS_PATH,
    tip_only: bool = False,
) -> int:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing PSM surface Gaussian asset: {path}. Run "
            "`python scripts/build_psm_surface_gaussians.py` first."
        )
    with np.load(path, allow_pickle=False) as asset:
        required = {
            "means",
            "quats_wxyz",
            "scales",
            "opacities",
            "colors",
            "link_ids",
            "link_names",
        }
        missing = sorted(required.difference(asset.files))
        if missing:
            raise KeyError(f"PSM Gaussian asset is missing arrays: {missing}")
        means = asset["means"].astype(np.float32)
        quats = asset["quats_wxyz"].astype(np.float32)
        scales = asset["scales"].astype(np.float32)
        opacities = asset["opacities"].astype(np.float32)
        colors = asset["colors"].astype(np.float32)
        link_ids = asset["link_ids"].astype(np.int64)
        link_names = asset["link_names"].tolist()

    if tip_only:
        # The long tool_main shaft dominates both the point count and the view.
        # The six remaining bodies are the wrist, wrist shafts and two jaws—the
        # portion whose tracked image pose the user actually needs.
        main_link_name = "PSM1_tool_main_link"
        if main_link_name not in link_names:
            raise KeyError(f"Missing expected PSM main link: {main_link_name}")
        main_link_id = link_names.index(main_link_name)
        keep = link_ids != main_link_id
        means = means[keep]
        quats = quats[keep]
        scales = scales[keep]
        opacities = opacities[keep]
        colors = colors[keep]
        link_ids = link_ids[keep]

    count = len(means)
    expected_shapes = {
        "quats_wxyz": (count, 4),
        "scales": (count, 3),
        "opacities": (count,),
        "colors": (count, 3),
        "link_ids": (count,),
    }
    arrays = {
        "quats_wxyz": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "link_ids": link_ids,
    }
    if means.shape != (count, 3):
        raise ValueError(f"Invalid PSM Gaussian means shape: {means.shape}")
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"Invalid PSM Gaussian {name} shape: {arrays[name].shape}, "
                f"expected {expected_shape}"
            )
    if not all(np.all(np.isfinite(array)) for array in (means, quats, scales, opacities, colors)):
        raise ValueError("PSM Gaussian asset contains non-finite values")
    if np.any(scales <= 0.0):
        raise ValueError("PSM Gaussian scales must be positive")
    if np.any(link_ids < 0) or np.any(link_ids >= len(link_names)):
        raise ValueError("PSM Gaussian asset contains invalid link ids")

    body_by_name = {
        name: body_id
        for body_id, name in enumerate(builder.body_name[:robot_body_count])
    }
    missing_links = [name for name in link_names if name not in body_by_name]
    if missing_links:
        raise KeyError(f"PSM Gaussian links are absent from the Warp model: {missing_links}")
    body_ids = np.asarray(
        [body_by_name[link_names[link_id]] for link_id in link_ids], dtype=np.int32
    )

    builder.gaussian_means.extend(means.tolist())
    builder.gaussian_quats.extend(quats.tolist())
    builder.gaussian_scales.extend(scales.tolist())
    builder.gaussian_opacities.extend(opacities.tolist())
    builder.gaussian_colors.extend(colors.tolist())
    builder.gaussian_body_ids.extend(body_ids.tolist())
    print(
        "[super_embodied] PSM visual surface gaussians: "
        f"{count} across {len(link_names)} links, "
        f"tip_only={tip_only}, "
        f"scale={scales.min() * 1e3:.2f}..{scales.max() * 1e3:.2f}mm"
    )
    return count


def load_psm_lnd_pose_driver(
    X_table_camera: np.ndarray,
    path: Path = PSM_LND_POSE_DRIVER_PATH,
    return_q7: bool = False,
) -> (
    tuple[np.ndarray, list[str], np.ndarray]
    | tuple[np.ndarray, list[str], np.ndarray, np.ndarray | None]
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing PSM pose driver: {path}. Build the selected driver first."
        )
    with np.load(path, allow_pickle=False) as asset:
        timestamps = asset["timestamps"].astype(np.float64)
        link_names = asset["link_names"].tolist()
        q7 = asset["q7"].astype(np.float64) if "q7" in asset.files else None
        if "poses_gui_world_xyz_xyzw" in asset.files:
            coordinate_frame = str(asset["coordinate_frame"].item())
            if coordinate_frame != "current_gui_table_world":
                raise ValueError(
                    f"Unexpected raw PSM coordinate frame: {coordinate_frame}"
                )
            saved_X_table_camera = asset[
                "X_gui_world_rectified_left_camera"
            ].astype(np.float64)
            coordinate_error = float(
                np.max(
                    np.abs(
                        saved_X_table_camera
                        - np.asarray(X_table_camera, dtype=np.float64)
                    )
                )
            )
            if coordinate_error > 1.0e-6:
                raise ValueError(
                    "Raw PSM driver was built for a different GUI coordinate "
                    f"frame (max transform error {coordinate_error:.3e})"
                )
            poses_table = asset[
                "poses_gui_world_xyz_xyzw"
            ].astype(np.float32)
        else:
            poses_rect = asset[
                "poses_rect_camera_xyz_xyzw"
            ].astype(np.float64)
            poses_table = np.empty_like(poses_rect, dtype=np.float32)
            for state_index in range(len(timestamps)):
                for link_index in range(len(link_names)):
                    pose = poses_rect[state_index, link_index]
                    T_rect_link = np.eye(4, dtype=np.float64)
                    T_rect_link[:3, :3] = Rotation.from_quat(
                        pose[3:]
                    ).as_matrix()
                    T_rect_link[:3, 3] = pose[:3]
                    T_table_link = X_table_camera @ T_rect_link
                    poses_table[state_index, link_index, :3] = (
                        T_table_link[:3, 3]
                    )
                    poses_table[state_index, link_index, 3:] = (
                        Rotation.from_matrix(
                            T_table_link[:3, :3]
                        ).as_quat()
                    )
    if poses_table.shape != (len(timestamps), len(link_names), 7):
        raise ValueError(f"Invalid PSM pose driver shape: {poses_table.shape}")
    if q7 is not None and q7.shape != (len(timestamps), 7):
        raise ValueError(f"Invalid PSM q7 shape in pose driver: {q7.shape}")
    if return_q7:
        return timestamps, link_names, poses_table, q7
    return timestamps, link_names, poses_table


def _poses_to_matrices(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Expected poses shape (N,7), got {poses.shape}")
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
    matrices[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix()
    matrices[:, :3, 3] = poses[:, :3]
    return matrices


def _matrices_to_poses(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError(f"Expected matrices shape (N,4,4), got {matrices.shape}")
    poses = np.empty((len(matrices), 7), dtype=np.float32)
    poses[:, :3] = matrices[:, :3, 3]
    poses[:, 3:] = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    return poses


PAPER_LND_T5_MESH = np.asarray(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _paper_lnd_component_transforms(
    joints: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Exact component FK used by the upstream paper-LND implementation."""
    theta0, theta1, theta2, theta3 = np.asarray(
        joints, dtype=np.float64
    )
    T45 = np.asarray(
        [
            [np.sin(theta0), np.cos(theta0), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [np.cos(theta0), -np.sin(theta0), 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T56 = np.asarray(
        [
            [np.sin(theta1), np.cos(theta1), 0.0, 0.0091],
            [0.0, 0.0, 1.0, 0.0],
            [np.cos(theta1), -np.sin(theta1), 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T4mesh = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T5mesh = np.asarray(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T7mesh = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
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


def apply_psm_pose_driver_joint_offsets(
    poses_table: np.ndarray,
    q7: np.ndarray,
    kinematics: PSMLNDKinematics,
    joint_offsets: np.ndarray,
) -> np.ndarray:
    """Add validated manual joint offsets without discarding a pose driver.

    ``poses_table`` may come from strict LND, registered LND, or corrected.  We
    apply the URDF joint origin/axis in the tracked parent-link frame.  Thus
    the driver remains the baseline and only the requested rigid joint motion
    is added on top of it.  In particular, jaw opening is always a symmetric
    rotation about the shared parent z axis; CAD registration is never allowed
    to tilt that hinge axis.
    """
    offsets = np.asarray(joint_offsets, dtype=np.float64)
    if offsets.shape != (7,):
        raise ValueError(f"Expected PSM joint_offsets shape (7,), got {offsets.shape}")
    paper_wrist_pitch = kinematics.manual_offset_conventions.get(
        "paper_wrist_pitch", {}
    )
    paper_wrist_pitch_enabled = (
        paper_wrist_pitch.get("model") == "paper_lnd_exact_component_fk"
    )
    unsupported = offsets.copy()
    unsupported[[3, 6]] = 0.0
    if paper_wrist_pitch_enabled:
        unsupported[4] = 0.0
    if not np.allclose(unsupported, 0.0):
        raise ValueError(
            "Manual pose-driver offsets support roll/jaw and the validated "
            "paper-LND wrist-pitch control only"
        )
    supported_indices = [3, 6]
    if paper_wrist_pitch_enabled:
        supported_indices.append(4)
    if np.allclose(offsets[supported_indices], 0.0):
        return np.asarray(poses_table, dtype=np.float32).copy()

    names = tuple(kinematics.link_names)
    required_names = (
        "PSM1_tool_main_link",
        "PSM1_tool_wrist_link",
        "PSM1_tool_wrist_sca_shaft_link",
        "PSM1_tool_wrist_sca_ee_link_1",
        "PSM1_tool_wrist_sca_ee_link_2",
    )
    missing = [name for name in required_names if name not in names]
    if missing:
        raise KeyError(f"PSM pose driver is missing links needed for GUI offsets: {missing}")

    selected = _poses_to_matrices(poses_table)
    if len(selected) != len(names):
        raise ValueError(
            f"Pose/link count mismatch: {len(selected)} poses for {len(names)} links"
        )
    q7 = np.asarray(q7, dtype=np.float64)
    if q7.shape != (7,):
        raise ValueError(f"Expected q7 shape (7,), got {q7.shape}")

    def rotation_axis(
        axis_xyz: tuple[float, float, float] | list[float],
        angle: float,
    ) -> np.ndarray:
        axis = np.asarray(axis_xyz, dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        rotation = np.eye(4, dtype=np.float64)
        rotation[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        return rotation

    def tracked_world_delta(
        parent_index: int,
        angle: float,
        joint_origin_xyz: tuple[float, float, float],
        joint_axis_xyz: tuple[float, float, float] | list[float],
    ) -> np.ndarray:
        joint_origin = np.eye(4, dtype=np.float64)
        joint_origin[:3, 3] = joint_origin_xyz
        parent_local_delta = (
            joint_origin
            @ rotation_axis(joint_axis_xyz, angle)
            @ np.linalg.inv(joint_origin)
        )
        return (
            selected[parent_index]
            @ parent_local_delta
            @ np.linalg.inv(selected[parent_index])
        )

    roll_offset = float(offsets[3])
    if not np.isclose(roll_offset, 0.0):
        roll_spec = kinematics.manual_offset_conventions.get(
            "roll",
            {
                "parent_link": "PSM1_tool_main_link",
                "origin_xyz": [0.0, 0.0, 0.4162],
                "axis_xyz": [0.0, 0.0, 1.0],
            },
        )
        main_index = names.index(str(roll_spec["parent_link"]))
        roll_world_delta = tracked_world_delta(
            main_index,
            roll_offset,
            tuple(roll_spec["origin_xyz"]),
            roll_spec["axis_xyz"],
        )
        for link_index, name in enumerate(names):
            if name != "PSM1_tool_main_link":
                selected[link_index] = roll_world_delta @ selected[link_index]

    wrist_pitch_offset = float(offsets[4])
    if not np.isclose(wrist_pitch_offset, 0.0):
        if not paper_wrist_pitch_enabled:
            raise ValueError(
                "The selected pose driver has no validated paper wrist control"
            )
        anchor_index = names.index(
            str(
                paper_wrist_pitch.get(
                    "anchor_link", "PSM1_tool_wrist_link"
                )
            )
        )
        joints = np.asarray(
            [
                q7[4] + wrist_pitch_offset,
                q7[5],
                0.5 * q7[6],
                0.5 * q7[6],
            ],
            dtype=np.float64,
        )
        components = _paper_lnd_component_transforms(joints)
        local_transforms = {
            "PSM1_tool_wrist_shaft_link": (
                components[1] @ np.linalg.inv(PAPER_LND_T5_MESH)
            ),
            "PSM1_tool_wrist_sca_link": components[2],
            "PSM1_tool_wrist_sca_shaft_link": components[1],
            "PSM1_tool_wrist_sca_ee_link_1": components[3],
            "PSM1_tool_wrist_sca_ee_link_2": components[4],
        }
        anchor = selected[anchor_index]
        for name, local_transform in local_transforms.items():
            selected[names.index(name)] = anchor @ local_transform

    jaw_offset = float(offsets[6])
    if not np.isclose(jaw_offset, 0.0):
        jaw_spec = kinematics.manual_offset_conventions.get(
            "jaw",
            {
                "parent_link": "PSM1_tool_wrist_sca_shaft_link",
                "origin_xyz": [0.0, 0.0, 0.0],
                "axis_xyz": [0.0, 0.0, 1.0],
                "child_links": [
                    "PSM1_tool_wrist_sca_ee_link_1",
                    "PSM1_tool_wrist_sca_ee_link_2",
                ],
                "half_angle_signs": [1.0, -1.0],
            },
        )
        jaw_parent_index = names.index(str(jaw_spec["parent_link"]))
        jaw_indices = [
            names.index(str(name)) for name in jaw_spec["child_links"]
        ]
        for jaw_index, sign in zip(
            jaw_indices, jaw_spec["half_angle_signs"], strict=True
        ):
            jaw_world_delta = tracked_world_delta(
                jaw_parent_index,
                float(sign) * 0.5 * jaw_offset,
                tuple(jaw_spec["origin_xyz"]),
                jaw_spec["axis_xyz"],
            )
            selected[jaw_index] = jaw_world_delta @ selected[jaw_index]

    return _matrices_to_poses(selected)


def enforce_psm_tissue_kinematic_gap(
    env: EmbodiedGaussiansEnvironment,
) -> float:
    """Optionally retract a driven PSM; disabled to preserve the pose driver."""
    if (
        getattr(env, "super_tissue_mode", "rigid_v9") not in SOFT_TISSUE_MODES
        or not getattr(env, "super_psm_tissue_collisions_enabled", False)
        or not getattr(
            env, "super_psm_tissue_kinematic_gap_guard_enabled", False
        )
        or env.sim.triangle_skin_contact_projector is None
    ):
        env.super_psm_tissue_kinematic_guard_offset_m = 0.0  # type: ignore[attr-defined]
        env.super_psm_tissue_kinematic_guard_gap_m = None  # type: ignore[attr-defined]
        return 0.0

    minimum_gap = float(
        getattr(
            env,
            "super_psm_tissue_kinematic_min_gap_m",
            ADAPTIVE_TISSUE_KINEMATIC_MIN_GAP_M,
        )
    )
    settings = env.physics_settings
    projector = env.sim.triangle_skin_contact_projector
    total_offset = 0.0
    previous_gap = -float("inf")
    for _ in range(4):
        projector.detect(
            env.sim.model,
            env.sim.state_0,
            settings.dt / settings.substeps,
            contact_margin_m=settings.triangle_skin_contact_margin_m,
            query_distance_m=settings.triangle_skin_query_distance_m,
            ccd_velocity_scale=0.0,
            friction_coefficient=0.0,
            relaxation=1.0,
        )
        gap = float(projector.metrics()["minimum_signed_distance_m"])
        env.super_psm_tissue_kinematic_guard_gap_m = gap  # type: ignore[attr-defined]
        if gap >= minimum_gap or gap <= previous_gap + 1.0e-8:
            break
        previous_gap = gap
        correction = min(minimum_gap - gap, 0.003)
        for body_ids in env.super_psm_lnd_body_ids:  # type: ignore[attr-defined]
            for state in (env.sim.state_0, env.sim.state_1):
                body_q = warp.to_torch(state.body_q)
                body_q[body_ids, 2] += correction
        total_offset += correction
    env.super_psm_tissue_kinematic_guard_offset_m = total_offset  # type: ignore[attr-defined]
    return total_offset


def apply_psm_lnd_pose(
    env: EmbodiedGaussiansEnvironment,
    state_index: int,
    joint_offsets: np.ndarray | None = None,
    translation_offset: np.ndarray | None = None,
    closure_blocked_by_tissue: bool = False,
    closing_requested: bool = False,
    update_gaussians: bool = True,
) -> None:
    poses = env.super_psm_lnd_poses_table  # type: ignore[attr-defined]
    body_ids_by_env = env.super_psm_lnd_body_ids  # type: ignore[attr-defined]
    state_index = max(0, min(int(state_index), len(poses) - 1))
    selected_poses = poses[state_index].copy()
    raw_q7 = np.asarray(
        env.super_psm_q7_states[state_index], dtype=np.float64  # type: ignore[attr-defined]
    )
    effective_joint_offsets = (
        np.zeros(7, dtype=np.float64)
        if joint_offsets is None
        else np.asarray(joint_offsets, dtype=np.float64).copy()
    )
    if effective_joint_offsets.shape != (7,):
        raise ValueError(
            "PSM joint_offsets must contain seven values"
        )
    jaw_angle_rad = float(raw_q7[6] + effective_joint_offsets[6])
    if not np.allclose(effective_joint_offsets, 0.0):
        selected_poses = apply_psm_pose_driver_joint_offsets(
            selected_poses,
            raw_q7,
            env.super_psm_lnd_kinematics,  # type: ignore[attr-defined]
            effective_joint_offsets,
        )
    env.sim.set_triangle_skin_jaw_signal(
        jaw_angle_rad,
        float(
            env.super_psm_lnd_timestamps[state_index]  # type: ignore[attr-defined]
        ),
        contact_limited_actuator_enabled=False,
        closure_blocked_by_tissue=closure_blocked_by_tissue,
        closing_requested=closing_requested,
    )
    if translation_offset is not None:
        translation = np.asarray(translation_offset, dtype=np.float64)
        if translation.shape != (3,):
            raise ValueError(
                f"Expected PSM translation_offset shape (3,), got {translation.shape}"
            )
        selected_poses = selected_poses.copy()
        selected_poses[:, :3] += translation[None, :]
    pose_tensor = torch.as_tensor(selected_poses, dtype=torch.float32)
    velocity_tensor = torch.as_tensor(
        env.super_psm_lnd_body_velocities[state_index], dtype=torch.float32  # type: ignore[attr-defined]
    )
    for body_ids in body_ids_by_env:
        for state in (env.sim.state_0, env.sim.state_1):
            body_q = warp.to_torch(state.body_q)
            body_q[body_ids] = pose_tensor.to(body_q.device)
            body_qd = warp.to_torch(state.body_qd)
            body_qd[body_ids] = velocity_tensor.to(body_qd.device)
    enforce_psm_tissue_kinematic_gap(env)
    if update_gaussians:
        env.sim.update_gaussian_transforms()


def set_psm_tissue_collisions(
    env: EmbodiedGaussiansEnvironment, enabled: bool
) -> bool:
    """Enable the configured tissue surface constraint."""
    tissue_mode = getattr(env, "super_tissue_mode", "rigid_v9")
    if tissue_mode in SOFT_TISSUE_MODES:
        if (
            enabled
            and env.sim.triangle_skin_contact_projector is None
        ):
            raise RuntimeError(
                "Soft tissue has no configured triangle-skin contact projector"
            )
        if enabled:
            settings = env.physics_settings
            projector = env.sim.triangle_skin_contact_projector
            projector.detect(
                env.sim.model,
                env.sim.state_0,
                settings.dt / settings.substeps,
                contact_margin_m=settings.triangle_skin_contact_margin_m,
                query_distance_m=settings.triangle_skin_query_distance_m,
                ccd_velocity_scale=0.0,
                friction_coefficient=0.0,
                relaxation=1.0,
            )
            metrics = projector.metrics()
            minimum_gap = metrics["minimum_signed_distance_m"]
            if minimum_gap < 0.0:
                print(
                    "[super_embodied] triangle-skin contact enabled from overlap: "
                    f"minimum_gap={minimum_gap * 1e3:+.3f} mm, "
                    f"candidates={metrics['contact_count']}; "
                    "the two jaw triangle surfaces will deform the local "
                    "tissue patch without changing the PSM pose."
                )
            else:
                print(
                    f"[super_embodied] {tissue_mode} triangle-skin contact enabled: "
                    f"minimum_gap={minimum_gap * 1e3:+.3f} mm, "
                    f"candidates={metrics['contact_count']}"
                )
        elif env.sim.triangle_skin_contact_projector is not None:
            env.sim.triangle_skin_contact_projector.reset_persistent_grip()
        warp.synchronize()
        env.physics_settings.enable_triangle_skin_contacts = bool(enabled)
        env.physics_settings.enable_particle_shape_contacts = False
        env.physics_settings.enable_particle_particle_contacts = False
        env.super_psm_tissue_collisions_enabled = bool(enabled)  # type: ignore[attr-defined]
        env.super_psm_collisions_enabled = bool(enabled)  # type: ignore[attr-defined]
        if hasattr(env.sim, "_physics_step_cache"):
            delattr(env.sim, "_physics_step_cache")
        return bool(enabled)

    # v9 fallback: preserve the historical rigid shape-pair switch.
    pair_count = int(
        getattr(env, "super_psm_tissue_contact_pair_count", 0)
    )
    if enabled and pair_count <= 0:
        raise RuntimeError("PSM/tissue contact pairs were not preallocated")
    warp.synchronize()
    env.sim.model.shape_contact_pair_count = pair_count if enabled else 0
    env.sim.model.rigid_contact_count.zero_()
    env.super_psm_tissue_collisions_enabled = bool(enabled)  # type: ignore[attr-defined]
    env.super_psm_collisions_enabled = bool(enabled)  # type: ignore[attr-defined]
    # The broad-phase launch dimension is captured in the physics CUDA graph.
    if hasattr(env.sim, "_physics_step_cache"):
        delattr(env.sim, "_physics_step_cache")
    return bool(enabled)


def build_environment(
    num_envs: int = 1,
    add_gaussians: bool = True,
    device: str = "cuda",
    psm_pose_driver_path: Path = PSM_LND_POSE_DRIVER_PATH,
    psm_visual_tip_only: bool = False,
    tissue_mode: str = "paper_pbd",
    tissue_asset_path_override: Path | None = None,
    psm_visual_enabled: bool = True,
    include_scene_background: bool = True,
) -> EmbodiedGaussiansEnvironment:
    if tissue_mode not in {*SOFT_TISSUE_MODES, "rigid_v9"}:
        raise ValueError(f"Unknown SUPER tissue mode: {tissue_mode}")
    if tissue_mode in SOFT_TISSUE_MODES and num_envs != 1:
        raise ValueError(
            "SUPER tetrahedral soft tissue currently supports num_envs=1"
        )
    is_soft_tissue = tissue_mode in SOFT_TISSUE_MODES
    if tissue_mode in PAPER_TISSUE_MODES:
        tissue_young_modulus_pa = PAPER_SOFT_TISSUE_YOUNG_MODULUS_PA
        tissue_poisson_ratio = PAPER_SOFT_TISSUE_POISSON_RATIO
        tissue_gravity_m_s2 = PAPER_SOFT_TISSUE_GRAVITY_M_S2
        tissue_velocity_damping = (
            PAPER_SOFT_TISSUE_VELOCITY_DAMPING_PER_SECOND
        )
        tissue_material_min_volume_ratio = (
            PAPER_SOFT_TISSUE_MATERIAL_MIN_VOLUME_RATIO
        )
        tissue_constraint_model = (
            "paper" if tissue_mode == "paper_pbd" else "neo_hookean"
        )
    elif tissue_mode == "adaptive_soft":
        tissue_young_modulus_pa = ADAPTIVE_TISSUE_YOUNG_MODULUS_PA
        tissue_poisson_ratio = ADAPTIVE_TISSUE_POISSON_RATIO
        tissue_gravity_m_s2 = ADAPTIVE_TISSUE_GRAVITY_M_S2
        tissue_velocity_damping = (
            ADAPTIVE_TISSUE_VELOCITY_DAMPING_PER_SECOND
        )
        tissue_material_min_volume_ratio = (
            ADAPTIVE_TISSUE_MATERIAL_MIN_VOLUME_RATIO
        )
        tissue_constraint_model = "neo_hookean"
    else:
        tissue_young_modulus_pa = None
        tissue_poisson_ratio = None
        tissue_gravity_m_s2 = -9.80665
        tissue_velocity_damping = None
        tissue_material_min_volume_ratio = None
        tissue_constraint_model = "none"
    resolved_pose_driver = Path(psm_pose_driver_path).resolve()
    if resolved_pose_driver == PSM_RAW_POSE_DRIVER_PATH.resolve():
        psm_urdf_path = PSM_RAW_URDF_PATH
        psm_mimic_map_path = PSM_RAW_MIMIC_MAP_PATH
        psm_surface_gaussians_path = PSM_RAW_SURFACE_GAUSSIANS_PATH
        psm_lnd_model_path = PSM_RAW_LND_MODEL_PATH
        psm_pose_report_path = PSM_RAW_REGISTRATION_REPORT_PATH
        psm_pose_source = "raw_kinematics"
        raw_kinematics_mode = True
    elif (
        resolved_pose_driver
        == PSM_RAW_P420006_POSE_DRIVER_PATH.resolve()
    ):
        psm_urdf_path = PSM_RAW_P420006_URDF_PATH
        psm_mimic_map_path = PSM_RAW_P420006_MIMIC_MAP_PATH
        psm_surface_gaussians_path = (
            PSM_RAW_P420006_SURFACE_GAUSSIANS_PATH
        )
        psm_lnd_model_path = PSM_RAW_LND_MODEL_PATH
        psm_pose_report_path = (
            PSM_RAW_P420006_REGISTRATION_REPORT_PATH
        )
        psm_pose_source = "raw_p420006"
        raw_kinematics_mode = True
    elif (
        resolved_pose_driver
        == PSM_RAW_P420006_STEREO_VISUAL_POSE_DRIVER_PATH.resolve()
    ):
        # Visual calibration changes only the timestamped pose driver.  CAD,
        # Gaussian appearance, contact geometry and the raw LND model remain
        # exactly the same as the raw P420006 version.
        psm_urdf_path = PSM_RAW_P420006_URDF_PATH
        psm_mimic_map_path = PSM_RAW_P420006_MIMIC_MAP_PATH
        psm_surface_gaussians_path = (
            PSM_RAW_P420006_SURFACE_GAUSSIANS_PATH
        )
        psm_lnd_model_path = PSM_RAW_LND_MODEL_PATH
        psm_pose_report_path = (
            PSM_RAW_P420006_REGISTRATION_REPORT_PATH
        )
        psm_pose_source = "raw_p420006_stereo_visual"
        raw_kinematics_mode = True
    elif (
        resolved_pose_driver
        == PSM_RAW_P420006_SAM2_ONLINE_POSE_DRIVER_PATH.resolve()
    ):
        # Full first-frame-only SurgicalSAM2 online correction.  Geometry and
        # the raw q7/LND source remain the same fixed P420006 backbone.
        psm_urdf_path = PSM_RAW_P420006_URDF_PATH
        psm_mimic_map_path = PSM_RAW_P420006_MIMIC_MAP_PATH
        psm_surface_gaussians_path = (
            PSM_RAW_P420006_SURFACE_GAUSSIANS_PATH
        )
        psm_lnd_model_path = PSM_RAW_LND_MODEL_PATH
        psm_pose_report_path = (
            PSM_RAW_P420006_REGISTRATION_REPORT_PATH
        )
        psm_pose_source = "raw_p420006_sam2_online"
        raw_kinematics_mode = True
    elif (
        resolved_pose_driver
        == PSM_RAW_PAPER_LND_SAM2_ONLINE_POSE_DRIVER_PATH.resolve()
    ):
        # Exact paper-repository LND meshes and component FK, corrected from a
        # fresh first-pair-only bilateral SurgicalSAM2 sequence.
        psm_urdf_path = PSM_RAW_PAPER_LND_URDF_PATH
        psm_mimic_map_path = PSM_RAW_PAPER_LND_MIMIC_MAP_PATH
        psm_surface_gaussians_path = (
            PSM_RAW_PAPER_LND_SURFACE_GAUSSIANS_PATH
        )
        psm_lnd_model_path = PSM_RAW_LND_MODEL_PATH
        psm_pose_report_path = (
            PSM_RAW_PAPER_LND_REGISTRATION_REPORT_PATH
        )
        psm_pose_source = "raw_paper_lnd_sam2_online"
        raw_kinematics_mode = True
    elif (
        resolved_pose_driver
        == PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_POSE_DRIVER_PATH.resolve()
    ):
        # Pair 0 supplies one shared binocular camera registration and one q5
        # zero offset. Every later state is raw q7 + exact paper component FK;
        # there is no time-varying visual residual or filter.
        psm_urdf_path = PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_URDF_PATH
        psm_mimic_map_path = (
            PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_MIMIC_MAP_PATH
        )
        psm_surface_gaussians_path = (
            PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_SURFACE_GAUSSIANS_PATH
        )
        psm_lnd_model_path = PSM_RAW_LND_MODEL_PATH
        psm_pose_report_path = (
            PSM_RAW_PAPER_LND_FIRST_STEREO_STATIC_REGISTRATION_REPORT_PATH
        )
        psm_pose_source = "raw_paper_lnd_first_stereo_static_q5"
        raw_kinematics_mode = True
    elif (
        resolved_pose_driver
        == PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_POSE_DRIVER_PATH.resolve()
    ):
        # Pair 0 supplies one shared binocular 6DoF camera registration.
        # Every raw q7 value, including the time-varying q5, is preserved
        # exactly; no later frame reads any visual correction.
        psm_urdf_path = PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_URDF_PATH
        psm_mimic_map_path = (
            PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_MIMIC_MAP_PATH
        )
        psm_surface_gaussians_path = (
            PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_SURFACE_GAUSSIANS_PATH
        )
        psm_lnd_model_path = PSM_RAW_LND_MODEL_PATH
        psm_pose_report_path = (
            PSM_RAW_PAPER_LND_FIRST_STEREO_SE3_FIXED_Q5_REGISTRATION_REPORT_PATH
        )
        psm_pose_source = "raw_paper_lnd_first_stereo_se3_fixed_q5"
        raw_kinematics_mode = True
    elif (
        resolved_pose_driver
        == (
            PSM_RAW_PAPER_LND_SAM2_MULTIANCHOR_CLOSEDJAW_POSE_DRIVER_PATH
        ).resolve()
    ):
        # Exact paper LND CAD/FK with multi-anchor bilateral SAM2.  Visual
        # correction is forbidden from changing q7, so the raw middle-sequence
        # jaw closure is preserved exactly.
        psm_urdf_path = (
            PSM_RAW_PAPER_LND_MULTIANCHOR_CLOSEDJAW_URDF_PATH
        )
        psm_mimic_map_path = (
            PSM_RAW_PAPER_LND_MULTIANCHOR_CLOSEDJAW_MIMIC_MAP_PATH
        )
        psm_surface_gaussians_path = (
            PSM_RAW_PAPER_LND_MULTIANCHOR_CLOSEDJAW_SURFACE_GAUSSIANS_PATH
        )
        psm_lnd_model_path = PSM_RAW_LND_MODEL_PATH
        psm_pose_report_path = (
            PSM_RAW_PAPER_LND_MULTIANCHOR_CLOSEDJAW_REGISTRATION_REPORT_PATH
        )
        psm_pose_source = "raw_paper_lnd_sam2_multianchor_closedjaw"
        raw_kinematics_mode = True
    elif (
        resolved_pose_driver
        == (
            PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_CLOSEDJAW_POSE_DRIVER_PATH
        ).resolve()
    ):
        # Exact paper LND CAD/FK with denser bilateral SAM2 anchors through
        # the tissue-contact interval. Visual correction cannot change q7.
        psm_urdf_path = (
            PSM_RAW_PAPER_LND_DENSE_CONTACT_CLOSEDJAW_URDF_PATH
        )
        psm_mimic_map_path = (
            PSM_RAW_PAPER_LND_DENSE_CONTACT_CLOSEDJAW_MIMIC_MAP_PATH
        )
        psm_surface_gaussians_path = (
            PSM_RAW_PAPER_LND_DENSE_CONTACT_CLOSEDJAW_SURFACE_GAUSSIANS_PATH
        )
        psm_lnd_model_path = PSM_RAW_LND_MODEL_PATH
        psm_pose_report_path = (
            PSM_RAW_PAPER_LND_DENSE_CONTACT_CLOSEDJAW_REGISTRATION_REPORT_PATH
        )
        psm_pose_source = "raw_paper_lnd_sam2_dense_contact_closedjaw"
        raw_kinematics_mode = True
    elif (
        resolved_pose_driver
        == (
            PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_SE3_ONLY_POSE_DRIVER_PATH
        ).resolve()
    ):
        # All 1631 strict stereo pairs contribute a time-varying SE(3)
        # residual. Every raw q7 column remains bitwise unchanged.
        psm_urdf_path = PSM_RAW_PAPER_LND_DENSE_CONTACT_SE3_ONLY_URDF_PATH
        psm_mimic_map_path = (
            PSM_RAW_PAPER_LND_DENSE_CONTACT_SE3_ONLY_MIMIC_MAP_PATH
        )
        psm_surface_gaussians_path = (
            PSM_RAW_PAPER_LND_DENSE_CONTACT_SE3_ONLY_SURFACE_GAUSSIANS_PATH
        )
        psm_lnd_model_path = PSM_RAW_LND_MODEL_PATH
        psm_pose_report_path = (
            PSM_RAW_PAPER_LND_DENSE_CONTACT_SE3_ONLY_REGISTRATION_REPORT_PATH
        )
        psm_pose_source = "raw_paper_lnd_sam2_dense_contact_se3_only"
        raw_kinematics_mode = True
    elif (
        resolved_pose_driver
        == (
            PSM_RAW_PAPER_LND_SAM2_DENSE_CONTACT_UNBOUNDED_XYZ_POSE_DRIVER_PATH
        ).resolve()
    ):
        # Twenty-two bilateral anchors, enhanced trustworthy manual jaw-tip
        # observations, unbounded camera-frame XYZ, bounded 3-D rotation, and
        # bitwise-exact raw q1..q7.
        psm_urdf_path = (
            PSM_RAW_PAPER_LND_DENSE_CONTACT_UNBOUNDED_XYZ_URDF_PATH
        )
        psm_mimic_map_path = (
            PSM_RAW_PAPER_LND_DENSE_CONTACT_UNBOUNDED_XYZ_MIMIC_MAP_PATH
        )
        psm_surface_gaussians_path = (
            PSM_RAW_PAPER_LND_DENSE_CONTACT_UNBOUNDED_XYZ_SURFACE_GAUSSIANS_PATH
        )
        psm_lnd_model_path = PSM_RAW_LND_MODEL_PATH
        psm_pose_report_path = (
            PSM_RAW_PAPER_LND_DENSE_CONTACT_UNBOUNDED_XYZ_REGISTRATION_REPORT_PATH
        )
        psm_pose_source = (
            "raw_paper_lnd_sam2_dense_contact_unbounded_xyz"
        )
        raw_kinematics_mode = True
    else:
        psm_urdf_path = PSM_URDF_PATH
        psm_mimic_map_path = PSM_MIMIC_MAP_PATH
        psm_surface_gaussians_path = PSM_SURFACE_GAUSSIANS_PATH
        psm_lnd_model_path = PSM_LND_MODEL_PATH
        psm_pose_report_path = PSM_LND_POSE_REPORT_PATH
        psm_pose_source = "legacy_derived"
        raw_kinematics_mode = False

    ground_data = read_ground(GROUND_PATH)
    ground = Ground(plane=ground_data)
    X_table_camera = load_table_from_camera()
    (
        lnd_timestamps,
        lnd_link_names,
        lnd_poses_table,
        driver_q7_states,
    ) = load_psm_lnd_pose_driver(
        X_table_camera,
        path=psm_pose_driver_path,
        return_q7=True,
    )
    if driver_q7_states is None:
        with open(ROBOTS_PATH, "r") as file:
            robot_data = json.load(file)["PSM1"]
        q7_states = np.asarray(
            [state["q"] for state in robot_data["states"]],
            dtype=np.float64,
        )
    else:
        q7_states = driver_q7_states
    if len(q7_states) != len(lnd_timestamps):
        raise ValueError(
            "PSM q7/pose-driver length mismatch: "
            f"{len(q7_states)} != {len(lnd_timestamps)}"
        )
    mimic_cfg = load_mimic_config(psm_mimic_map_path)
    joint_order = urdf_actuated_joint_order(psm_urdf_path)
    q_start = load_initial_psm_q_full(
        q7_states[0],
        mimic_cfg=mimic_cfg,
        joint_order=joint_order,
    )
    psm_X_WB = X_table_camera
    if raw_kinematics_mode:
        with np.load(psm_pose_driver_path, allow_pickle=False) as raw_driver:
            psm_X_WB = raw_driver[
                "X_gui_world_urdf_base"
            ].astype(np.float32)
    tissue_body = None
    soft_tissue = None
    tissue_shape_density = None
    if tissue_mode == "rigid_v9":
        scene_metadata = json.loads(
            SCENE_METADATA_PATH.read_text(encoding="utf-8")
        )
        tissue_shape_density = scene_metadata.get(
            "tissue_shape_density_kg_m3"
        )
        tissue_body = load_body(TISSUE_PATH)
        if tissue_body.gaussians is not None:
            for _i in range(len(tissue_body.gaussians.scales)):
                tissue_body.gaussians.scales[_i] = [
                    s * TISSUE_GAUSSIAN_SCALE
                    for s in tissue_body.gaussians.scales[_i]
                ]
        if tissue_body.particles is not None:
            tissue_body.particles.radii = [
                float(radius) * TISSUE_PARTICLE_RADIUS_SCALE
                for radius in tissue_body.particles.radii
            ]
    else:
        if tissue_asset_path_override is not None:
            tissue_asset_path = Path(tissue_asset_path_override).resolve()
        elif tissue_mode == "paper_pbd":
            tissue_asset_path = PAPER_CONSTRAINT_TISSUE_PATH
        elif tissue_mode == "paper_soft":
            tissue_asset_path = PAPER_PBD_TISSUE_PATH
        else:
            tissue_asset_path = ADAPTIVE_TISSUE_PATH
        if not tissue_asset_path.exists():
            raise FileNotFoundError(
                f"Missing {tissue_mode} tissue asset: {tissue_asset_path}"
            )
        soft_tissue = SoftBody.from_npz(
            tissue_asset_path, name=f"super_tissue_{tissue_mode}"
        )
        print(
            f"[super_embodied] {tissue_mode} asset={tissue_asset_path}; "
            f"particles={len(soft_tissue.tetra_mesh.rest_positions)}, "
            f"tetrahedra={len(soft_tissue.tetra_mesh.tet_indices)}, "
            f"skin_triangles={len(soft_tissue.tetra_mesh.surface_faces)}, "
            "depth_residual=OFF, stiffness_optimization=OFF"
        )
    ground_body = load_body(GROUND_BODY_PATH)
    if ground_body.gaussians is not None:
        for _i in range(len(ground_body.gaussians.scales)):
            ground_body.gaussians.scales[_i] = [
                s * GROUND_GAUSSIAN_SCALE for s in ground_body.gaussians.scales[_i]
            ]

    scene_gravity_m_s2 = tissue_gravity_m_s2
    builder = EmbodiedGaussiansBuilder(
        up_vector=ground.normal(),
        gravity=scene_gravity_m_s2,
    )
    builder.add_renderable_articulation_from_urdf(
        urdf_path=psm_urdf_path,
        initial_joints=q_start,
        X_WB=psm_X_WB,
        stiffness=500,
        damping=100,
        ignore_inertial_definitions=True,
        ensure_nonstatic_links=True,
        collapse_fixed_joints=False,
        # PSM appearance is loaded from the preprocessed URDF visual meshes below.
        add_gaussians=False,
    )
    robot_body_count = builder.body_count

    psm_gaussian_count = 0
    if add_gaussians and psm_visual_enabled:
        psm_gaussian_count = add_psm_surface_gaussians(
            builder,
            robot_body_count,
            path=psm_surface_gaussians_path,
            tip_only=psm_visual_tip_only,
        )

    # PSM is driven directly from the timestamped pose stream. Keep its rigid
    # collisions disabled so it cannot disturb the rigid scene reconstruction.
    psm_shape_ids = []
    psm_collision_shape_ids = []
    for shape_id, body_id in enumerate(builder.shape_body):
        if 0 <= body_id < robot_body_count:
            psm_shape_ids.append(shape_id)
            if builder.shape_shape_collision[shape_id]:
                psm_collision_shape_ids.append(shape_id)
            builder.shape_shape_collision[shape_id] = False
            builder.shape_ground_collision[shape_id] = False
    if any(builder.shape_shape_collision[i] for i in psm_shape_ids):
        raise RuntimeError("Failed to disable PSM shape collisions.")
    if any(builder.shape_ground_collision[i] for i in psm_shape_ids):
        raise RuntimeError("Failed to disable PSM ground collisions.")
    # Keep disabled PSM shapes out of Warp's universal group. Dense tissue
    # spheres use one group each to avoid quadratic same-body filter pairs.
    psm_collision_group = max(builder.last_collision_group + 1, 1)
    for shape_id in psm_shape_ids:
        old_group = builder.shape_collision_group[shape_id]
        builder.shape_collision_group_map[old_group].remove(shape_id)
        builder.shape_collision_group[shape_id] = psm_collision_group
    builder.shape_collision_group_map = {
        group: shape_ids
        for group, shape_ids in builder.shape_collision_group_map.items()
        if shape_ids
    }
    builder.shape_collision_group_map[psm_collision_group] = psm_shape_ids.copy()
    builder.last_collision_group = psm_collision_group
    print(
        "[super_embodied] PSM collisions disabled: "
        f"shape_shape=0/{len(psm_shape_ids)}, "
        f"shape_ground=0/{len(psm_shape_ids)}"
    )

    tissue_body_id = None
    soft_tissue_handle = None
    if tissue_mode == "rigid_v9":
        assert tissue_body is not None
        tissue_body_id = builder.add_rigid_body(
            tissue_body,
            mu=0.05,
            add_gaussians=add_gaussians,
            density=tissue_shape_density,
            individual_collision_groups=True,
        )
    else:
        assert soft_tissue is not None
        assert tissue_young_modulus_pa is not None
        assert tissue_poisson_ratio is not None
        soft_tissue_handle = builder.add_soft_body(
            soft_tissue,
            young_modulus_pa=tissue_young_modulus_pa,
            poisson_ratio=tissue_poisson_ratio,
            anchor_mode="support_candidate",
            add_gaussians=add_gaussians,
            add_collision_skin=True,
        )
    contact_body_ids = {
        body_id
        for body_id, body_name in enumerate(builder.body_name)
        if body_name in PSM_TISSUE_CONTACT_LINK_NAMES
    }
    missing_contact_links = set(PSM_TISSUE_CONTACT_LINK_NAMES).difference(
        builder.body_name
    )
    if missing_contact_links:
        raise KeyError(f"Missing PSM tissue-contact links: {sorted(missing_contact_links)}")
    local_psm_tissue_contact_shape_ids = [
        shape_id
        for shape_id in psm_collision_shape_ids
        if builder.shape_body[shape_id] in contact_body_ids
    ]
    local_contact_shapes_by_link = {
        link_name: [
            shape_id
            for shape_id in local_psm_tissue_contact_shape_ids
            if builder.body_name[builder.shape_body[shape_id]]
            == link_name
        ]
        for link_name in PSM_TISSUE_CONTACT_LINK_NAMES
    }
    local_jaw_contact_shape_ids = [
        local_contact_shapes_by_link[link_name][0]
        for link_name in PSM_TISSUE_JAW_CONTACT_LINK_NAMES
        if len(local_contact_shapes_by_link[link_name]) == 1
    ]
    local_tissue_shape_ids = (
        [
            shape_id
            for shape_id, body_id in enumerate(builder.shape_body)
            if body_id == tissue_body_id
        ]
        if tissue_body_id is not None
        else []
    )
    if not local_psm_tissue_contact_shape_ids:
        raise RuntimeError("No collision geometry found for the PSM gripper links")
    if len(local_jaw_contact_shape_ids) != 2:
        raise RuntimeError(
            "Jaw triangle contact requires one collision mesh on each jaw; "
            f"got {local_contact_shapes_by_link}"
        )
    if (
        tissue_mode == "rigid_v9"
        and tissue_body is not None
        and len(local_tissue_shape_ids) != len(tissue_body.particles.means)
    ):
        raise RuntimeError("Tissue collision-shape count does not match its particles")

    if add_gaussians and include_scene_background:
        builder.add_visual_body(ground_body)

    final_builder = EmbodiedGaussiansBuilder(
        up_vector=ground.normal(),
        gravity=scene_gravity_m_s2,
    )
    psm_base_body_ids: list[int] = []
    psm_body_ids: list[list[int]] = []
    psm_lnd_body_ids: list[list[int]] = []
    tissue_body_ids: list[int] = []
    soft_tissue_handles = []
    psm_tissue_contact_shape_ids: list[list[int]] = []
    psm_jaw_contact_shape_ids: list[list[int]] = []
    tissue_shape_ids: list[list[int]] = []
    for _ in range(num_envs):
        body_start = final_builder.body_count
        shape_start = final_builder.shape_count
        final_builder.add_builder(builder, separate_collision_group=False)
        psm_body_ids.append(
            list(range(body_start, body_start + robot_body_count))
        )
        if tissue_body_id is not None:
            tissue_body_ids.append(body_start + tissue_body_id)
        if soft_tissue_handle is not None:
            soft_tissue_handles.append(final_builder.soft_body_handles[-1])
        psm_tissue_contact_shape_ids.append(
            [shape_start + shape_id for shape_id in local_psm_tissue_contact_shape_ids]
        )
        psm_jaw_contact_shape_ids.append(
            [shape_start + shape_id for shape_id in local_jaw_contact_shape_ids]
        )
        tissue_shape_ids.append(
            [shape_start + shape_id for shape_id in local_tissue_shape_ids]
        )
        body_names = final_builder.body_name[body_start : final_builder.body_count]
        body_id_by_name = {
            body_name: body_start + local_body_id
            for local_body_id, body_name in enumerate(body_names)
        }
        missing_lnd_links = [
            name for name in lnd_link_names if name not in body_id_by_name
        ]
        if missing_lnd_links:
            raise KeyError(f"LND-driven PSM links missing from model: {missing_lnd_links}")
        psm_lnd_body_ids.append(
            [body_id_by_name[name] for name in lnd_link_names]
        )
        for local_body_id, body_name in enumerate(body_names):
            if body_name == PSM_BASE_LINK_NAME:
                psm_base_body_ids.append(body_start + local_body_id)

    # Warp's add_builder() copies the child builder's default ground settings,
    # so the dataset plane must be applied after all child builders are added.
    final_builder.set_ground_plane(ground.normal(), ground.offset())

    env = EmbodiedGaussiansEnvironment(final_builder, device=device)

    model = env.sim.model
    contact_pair_count = 0
    if tissue_mode == "rigid_v9":
        contact_pairs = np.concatenate(
            [
                np.column_stack(
                    (
                        np.repeat(psm_shapes, len(tissue_shapes)),
                        np.tile(tissue_shapes, len(psm_shapes)),
                    )
                )
                for psm_shapes, tissue_shapes in zip(
                    psm_tissue_contact_shape_ids,
                    tissue_shape_ids,
                    strict=True,
                )
            ],
            axis=0,
        ).astype(np.int32, copy=False)
        if model.shape_contact_pair_count != 0:
            raise RuntimeError(
                "Unexpected rigid contact pairs before PSM/tissue contact setup: "
                f"{model.shape_contact_pair_count}"
            )
        model.shape_contact_pairs = warp.array(
            contact_pairs, dtype=warp.int32, device=model.device
        )
        model.shape_contact_pair_count = len(contact_pairs)
        contact_max, limited_contact_max = model.count_contact_points()
        model.allocate_rigid_contacts(
            count=contact_max,
            limited_contact_count=limited_contact_max,
            requires_grad=model.requires_grad,
        )
        model.shape_contact_pair_count = 0
        contact_pair_count = len(contact_pairs)
    elif tissue_mode in SOFT_TISSUE_MODES:
        env.sim.configure_triangle_skin_contacts(
            psm_tissue_contact_shape_ids[0],
            sample_spacing_m=(
                PAPER_SOFT_TISSUE_TOOL_SAMPLE_SPACING_M
                if tissue_mode in PAPER_TISSUE_MODES
                else ADAPTIVE_TISSUE_TOOL_SAMPLE_SPACING_M
            ),
            spread_layers=(
                PAPER_SOFT_TISSUE_CONTACT_SPREAD_LAYERS
                if tissue_mode in PAPER_TISSUE_MODES
                else ADAPTIVE_TISSUE_CONTACT_SPREAD_LAYERS
            ),
            top_support_lateral_radius_m=(
                PAPER_SOFT_TISSUE_TOP_SUPPORT_RADIUS_M
                if tissue_mode in PAPER_TISSUE_MODES
                else ADAPTIVE_TISSUE_TOP_CONTACT_SUPPORT_RADIUS_M
            ),
            top_support_depth_m=(
                PAPER_SOFT_TISSUE_TOP_SUPPORT_DEPTH_M
                if tissue_mode in PAPER_TISSUE_MODES
                else ADAPTIVE_TISSUE_TOP_CONTACT_SUPPORT_DEPTH_M
            ),
            top_support_weight_scale=(
                PAPER_SOFT_TISSUE_TOP_SUPPORT_WEIGHT_SCALE
                if tissue_mode in PAPER_TISSUE_MODES
                else 0.15
            ),
            top_pressure_shoulder_lateral_radius_m=(
                PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_RADIUS_M
                if tissue_mode in PAPER_TISSUE_MODES
                else 0.0
            ),
            top_pressure_shoulder_depth_m=(
                PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_DEPTH_M
                if tissue_mode in PAPER_TISSUE_MODES
                else 0.0
            ),
            top_pressure_shoulder_upward_scale=(
                PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_UPWARD_SCALE
                if tissue_mode in PAPER_TISSUE_MODES
                else 0.0
            ),
            top_pressure_shoulder_outward_scale=(
                PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_OUTWARD_SCALE
                if tissue_mode in PAPER_TISSUE_MODES
                else 0.0
            ),
            top_pressure_shoulder_bias_direction_world=(
                PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_BIAS_DIRECTION_WORLD
                if tissue_mode in PAPER_TISSUE_MODES
                else (0.0, 0.0, 0.0)
            ),
            top_pressure_shoulder_bias_start_m=(
                PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_BIAS_START_M
            ),
            top_barrier_lateral_tolerance_m=(
                PAPER_SOFT_TISSUE_TOP_BARRIER_LATERAL_TOLERANCE_M
                if tissue_mode in PAPER_TISSUE_MODES
                else ADAPTIVE_TISSUE_TOP_CONTACT_SUPPORT_RADIUS_M
            ),
            top_barrier_contact_patch_radius_m=(
                PAPER_SOFT_TISSUE_TOP_BARRIER_CONTACT_PATCH_RADIUS_M
                if tissue_mode in PAPER_TISSUE_MODES
                else 0.0
            ),
            top_barrier_clearance_m=(
                PAPER_SOFT_TISSUE_TOP_CLEARANCE_M
                if tissue_mode in PAPER_TISSUE_MODES
                else ADAPTIVE_TISSUE_CONTACT_MARGIN_M
            ),
            top_barrier_shape_ids=(
                psm_jaw_contact_shape_ids[0]
                if tissue_mode in PAPER_TISSUE_MODES
                else []
            ),
            jaw_contact_shape_ids=psm_jaw_contact_shape_ids[0],
            jaw_contact_distal_length_m=(
                PAPER_SOFT_TISSUE_JAW_CONTACT_DISTAL_LENGTH_M
                if tissue_mode in PAPER_TISSUE_MODES
                else 0.0
            ),
            top_barrier_distal_length_m=(
                PAPER_SOFT_TISSUE_TOP_BARRIER_DISTAL_LENGTH_M
                if tissue_mode in PAPER_TISSUE_MODES
                else 0.0
            ),
            top_barrier_tip_allowance_m=(
                PAPER_SOFT_TISSUE_TOP_BARRIER_TIP_ALLOWANCE_M
                if tissue_mode in PAPER_TISSUE_MODES
                else 0.0
            ),
            jaw_friction_coefficient=(
                PAPER_SOFT_TISSUE_JAW_FRICTION_COEFFICIENT
            ),
            persistent_grip_enabled=(tissue_mode in PAPER_TISSUE_MODES),
            persistent_grip_minimum_contact_samples_per_jaw=(
                PAPER_SOFT_TISSUE_GRIP_MIN_CONTACT_SAMPLES_PER_JAW
            ),
            persistent_grip_nearest_surface_particles=(
                PAPER_SOFT_TISSUE_GRIP_NEAREST_SURFACE_PARTICLES
            ),
            persistent_grip_maximum_jaw_patch_separation_m=(
                PAPER_SOFT_TISSUE_GRIP_MAX_PATCH_SEPARATION_M
            ),
            persistent_grip_activation_steps=(
                PAPER_SOFT_TISSUE_GRIP_ACTIVATION_STEPS
            ),
            persistent_grip_maximum_capture_penetration_m=(
                PAPER_SOFT_TISSUE_GRIP_MAX_CAPTURE_PENETRATION_M
            ),
            persistent_grip_minimum_capture_volume_ratio=(
                PAPER_SOFT_TISSUE_GRIP_MIN_CAPTURE_VOLUME_RATIO
            ),
            persistent_grip_closed_angle_max_rad=(
                PAPER_SOFT_TISSUE_GRIP_CLOSED_ANGLE_MAX_RAD
            ),
            persistent_grip_release_angle_min_rad=(
                PAPER_SOFT_TISSUE_GRIP_RELEASE_ANGLE_MIN_RAD
            ),
            persistent_grip_release_angle_delta_rad=(
                PAPER_SOFT_TISSUE_GRIP_RELEASE_ANGLE_DELTA_RAD
            ),
            persistent_grip_wide_open_angle_rad=(
                PAPER_SOFT_TISSUE_GRIP_WIDE_OPEN_ANGLE_RAD
            ),
            persistent_grip_angle_motion_epsilon_rad=(
                PAPER_SOFT_TISSUE_GRIP_ANGLE_MOTION_EPSILON_RAD
            ),
            persistent_grip_compliance_m_per_n=(
                PAPER_SOFT_TISSUE_GRIP_COMPLIANCE_M_PER_N
            ),
            persistent_grip_relaxation=(
                PAPER_SOFT_TISSUE_GRIP_RELAXATION
            ),
            persistent_grip_maximum_correction_m=(
                PAPER_SOFT_TISSUE_GRIP_MAX_CORRECTION_M
            ),
            persistent_grip_transfer_layers=(
                PAPER_SOFT_TISSUE_GRIP_TRANSFER_LAYERS
            ),
            persistent_grip_minimum_volume_ratio=(
                PAPER_SOFT_TISSUE_CONTACT_MIN_VOLUME_RATIO
            ),
            persistent_grip_support_radius_m=(
                PAPER_SOFT_TISSUE_GRIP_SUPPORT_RADIUS_M
            ),
            persistent_grip_support_generations=(
                PAPER_SOFT_TISSUE_GRIP_SUPPORT_GENERATIONS
            ),
        )
        metrics = env.sim.triangle_skin_contact_metrics()
        print(
            f"[super_embodied] {tissue_mode} triangle-skin contact configured: "
            f"faces={len(final_builder.soft_collision_skin_faces)}, "
            f"tool_samples={metrics['sample_count'] if metrics else 0}, "
            "jaw_only=ON, shaft_contact=OFF, "
            f"jaw_top_plane={'ON' if tissue_mode in PAPER_TISSUE_MODES else 'OFF'}"
        )

    # The LND pose stream kinematically drives PSM links.
    gravity_factor = warp.to_torch(env.sim.model.gravity_factor).reshape(num_envs, -1)
    gravity_factor[:, :robot_body_count] = 0.0
    flat_psm_body_ids = [body_id for ids in psm_body_ids for body_id in ids]
    warp.to_torch(env.sim.model.body_inv_mass)[flat_psm_body_ids] = 0.0
    warp.to_torch(env.sim.model.body_inv_inertia)[flat_psm_body_ids] = 0.0

    q_start_t = torch.from_numpy(q_start).float()
    env.set_robot_q(PSM_ARTICULATION_INDEX, q_start_t)
    env.set_robot_desired_q(PSM_ARTICULATION_INDEX, q_start_t)
    env.super_tissue_mode = tissue_mode  # type: ignore[attr-defined]
    env.super_tissue_asset_path = (  # type: ignore[attr-defined]
        tissue_asset_path if is_soft_tissue else TISSUE_PATH
    )
    env.super_tissue_body_id = tissue_body_id  # type: ignore[attr-defined]
    env.super_tissue_body_ids = tissue_body_ids  # type: ignore[attr-defined]
    env.super_tissue_soft_handle = (  # type: ignore[attr-defined]
        soft_tissue_handles[0] if soft_tissue_handles else None
    )
    env.super_tissue_soft_handles = soft_tissue_handles  # type: ignore[attr-defined]
    env.super_tissue_young_modulus_pa = (  # type: ignore[attr-defined]
        tissue_young_modulus_pa
    )
    env.super_tissue_poisson_ratio = (  # type: ignore[attr-defined]
        tissue_poisson_ratio
    )
    env.super_tissue_gravity_m_s2 = (  # type: ignore[attr-defined]
        scene_gravity_m_s2 if is_soft_tissue else None
    )
    env.super_tissue_residual_mapping_enabled = False  # type: ignore[attr-defined]
    env.super_tissue_stiffness_optimization_enabled = False  # type: ignore[attr-defined]
    env.super_tissue_fixed_material_parameters = (  # type: ignore[attr-defined]
        tissue_mode in PAPER_TISSUE_MODES
    )
    env.super_tissue_constraint_model = tissue_constraint_model  # type: ignore[attr-defined]
    env.super_tissue_paper_stiffness = (  # type: ignore[attr-defined]
        {
            "distance": PAPER_CONSTRAINT_DISTANCE_STIFFNESS,
            "volume": PAPER_CONSTRAINT_VOLUME_STIFFNESS,
            "shape": PAPER_CONSTRAINT_SHAPE_STIFFNESS,
        }
        if tissue_mode == "paper_pbd"
        else None
    )
    env.super_robot_body_count = robot_body_count  # type: ignore[attr-defined]
    env.super_psm_gaussian_count = psm_gaussian_count  # type: ignore[attr-defined]
    env.super_psm_base_body_ids = psm_base_body_ids  # type: ignore[attr-defined]
    env.super_psm_body_ids = psm_body_ids  # type: ignore[attr-defined]
    env.super_psm_collisions_enabled = False  # type: ignore[attr-defined]
    env.super_psm_tissue_contact_shape_ids = (  # type: ignore[attr-defined]
        psm_tissue_contact_shape_ids
    )
    env.super_psm_tissue_contact_pair_count = contact_pair_count  # type: ignore[attr-defined]
    env.super_psm_tissue_collisions_enabled = False  # type: ignore[attr-defined]
    env.super_psm_tissue_kinematic_min_gap_m = (  # type: ignore[attr-defined]
        ADAPTIVE_TISSUE_KINEMATIC_MIN_GAP_M
    )
    env.super_psm_tissue_kinematic_gap_guard_enabled = (  # type: ignore[attr-defined]
        tissue_mode in SOFT_TISSUE_MODES
        and ADAPTIVE_TISSUE_ENABLE_KINEMATIC_GAP_GUARD
    )
    env.super_psm_tissue_kinematic_guard_offset_m = 0.0  # type: ignore[attr-defined]
    env.super_psm_tissue_kinematic_guard_gap_m = None  # type: ignore[attr-defined]
    env.super_psm_lnd_timestamps = lnd_timestamps  # type: ignore[attr-defined]
    env.super_psm_pose_driver_path = psm_pose_driver_path  # type: ignore[attr-defined]
    env.super_psm_pose_source = psm_pose_source  # type: ignore[attr-defined]
    env.super_psm_urdf_path = psm_urdf_path  # type: ignore[attr-defined]
    env.super_psm_mimic_map_path = (  # type: ignore[attr-defined]
        psm_mimic_map_path
    )
    env.super_psm_surface_gaussians_path = (  # type: ignore[attr-defined]
        psm_surface_gaussians_path
    )
    env.super_psm_lnd_link_names = lnd_link_names  # type: ignore[attr-defined]
    env.super_psm_lnd_poses_table = lnd_poses_table  # type: ignore[attr-defined]
    env.super_psm_lnd_body_ids = psm_lnd_body_ids  # type: ignore[attr-defined]
    body_com = warp.to_torch(env.sim.model.body_com)[psm_lnd_body_ids[0]].cpu().numpy()
    env.super_psm_lnd_body_velocities = compute_lnd_body_velocities(  # type: ignore[attr-defined]
        lnd_timestamps, lnd_poses_table, body_com
    )
    env.super_psm_q7_states = q7_states  # type: ignore[attr-defined]
    env.super_psm_lnd_kinematics = PSMLNDKinematics.from_files(  # type: ignore[attr-defined]
        psm_lnd_model_path,
        psm_pose_report_path,
        TABLE_FRAME_PATH,
        lnd_link_names,
    )
    visual_settings = env.visual_forces_settings
    visual_settings.lr_means = SUPER_VISUAL_FORCE_LR_MEANS
    visual_settings.kp = SUPER_VISUAL_FORCE_KP
    visual_settings.observations_are_bgr = True
    visual_settings.reset_optimizer_each_step = True
    visual_settings.normalize_forces_by_gaussian_count = True
    visual_settings.max_force = SUPER_VISUAL_FORCE_MAX_FORCE_N
    visual_settings.max_moment = SUPER_VISUAL_FORCE_MAX_MOMENT_NM
    visual_settings.robust_loss_beta = SUPER_VISUAL_FORCE_ROBUST_LOSS_BETA
    if is_soft_tissue:
        assert tissue_velocity_damping is not None
        assert tissue_material_min_volume_ratio is not None
        env.physics_settings.substeps = 12
        env.physics_settings.xpbd_iterations = 3
        env.physics_settings.use_project_material_tetrahedra = True
        env.physics_settings.tetrahedral_constraint_model = (
            tissue_constraint_model
        )
        env.physics_settings.paper_distance_stiffness = (
            PAPER_CONSTRAINT_DISTANCE_STIFFNESS
        )
        env.physics_settings.paper_volume_stiffness = (
            PAPER_CONSTRAINT_VOLUME_STIFFNESS
        )
        env.physics_settings.paper_shape_stiffness = (
            PAPER_CONSTRAINT_SHAPE_STIFFNESS
        )
        env.physics_settings.material_iterations = (
            PAPER_CONSTRAINT_MATERIAL_ITERATIONS
            if tissue_mode == "paper_pbd"
            else (
                PAPER_SOFT_TISSUE_MATERIAL_ITERATIONS
                if tissue_mode == "paper_soft"
                else 20
            )
        )
        env.physics_settings.material_relaxation = (
            PAPER_CONSTRAINT_MATERIAL_RELAXATION
            if tissue_mode == "paper_pbd"
            else (
                PAPER_SOFT_TISSUE_MATERIAL_RELAXATION
                if tissue_mode == "paper_soft"
                else 0.15
            )
        )
        env.physics_settings.material_compliance_scale = 1.0
        env.physics_settings.material_min_volume_ratio = (
            tissue_material_min_volume_ratio
        )
        # Constraint projection still changes geometry immediately, but only a
        # small fraction is treated as inertial velocity. This removes the
        # contact -> material -> velocity positive-feedback loop that made the
        # light tetrahedral nodes chatter around a kinematic jaw.
        env.physics_settings.material_projection_velocity_scale = (
            PAPER_SOFT_TISSUE_MATERIAL_PROJECTION_VELOCITY_SCALE
            if tissue_mode in PAPER_TISSUE_MODES
            else 1.0
        )
        # Kinematic jaw contact must retain its transferred velocity even
        # though distance/volume/shape recovery is strongly attenuated to
        # avoid chatter.  Keeping these scales separate makes q7 closure
        # visibly push tissue instead of being swallowed by the material pass.
        env.physics_settings.contact_projection_velocity_scale = (
            PAPER_SOFT_TISSUE_CONTACT_PROJECTION_VELOCITY_SCALE
            if tissue_mode in PAPER_TISSUE_MODES
            else 1.0
        )
        env.physics_settings.particle_velocity_damping_per_second = (
            tissue_velocity_damping
        )
        env.physics_settings.particle_ground_relaxation = 0.9
        env.physics_settings.enable_particle_shape_contacts = False
        env.physics_settings.enable_particle_particle_contacts = False
        env.physics_settings.enable_triangle_skin_contacts = False
        if tissue_mode in SOFT_TISSUE_MODES:
            # The geometric PSM/skin contact remains independent of depth
            # residual mapping and online stiffness optimization.
            # The dense skin query dominates contact cost. Solving it every
            # second substep gives six smaller contact updates per 60 Hz frame;
            # material projection still runs on all twelve substeps.
            env.physics_settings.triangle_skin_contact_substep_stride = (
                PAPER_SOFT_TISSUE_CONTACT_SUBSTEP_STRIDE
            )
            env.physics_settings.triangle_skin_contact_margin_m = (
                ADAPTIVE_TISSUE_CONTACT_MARGIN_M
            )
            env.physics_settings.triangle_skin_query_distance_m = (
                ADAPTIVE_TISSUE_CONTACT_QUERY_DISTANCE_M
            )
            env.physics_settings.triangle_skin_contact_relaxation = 1.0
            env.physics_settings.triangle_skin_contact_max_correction_m = (
                PAPER_SOFT_TISSUE_SURFACE_MAX_CORRECTION_M
                if tissue_mode in PAPER_TISSUE_MODES
                else ADAPTIVE_TISSUE_CONTACT_MAX_CORRECTION_M
            )
            env.physics_settings.triangle_skin_top_barrier_max_correction_m = (
                PAPER_SOFT_TISSUE_TOP_BARRIER_MAX_CORRECTION_M
                if tissue_mode in PAPER_TISSUE_MODES
                else 0.0
            )
            env.physics_settings.triangle_skin_contact_iterations = (
                PAPER_SOFT_TISSUE_SURFACE_ITERATIONS
                if tissue_mode in PAPER_TISSUE_MODES
                else 1
            )
            env.physics_settings.triangle_skin_post_contact_material_iterations = (
                PAPER_SOFT_TISSUE_POST_CONTACT_MATERIAL_ITERATIONS
                if tissue_mode in PAPER_TISSUE_MODES
                else 0
            )
            env.physics_settings.triangle_skin_final_barrier_max_correction_m = (
                PAPER_SOFT_TISSUE_FINAL_BARRIER_MAX_CORRECTION_M
                if tissue_mode in PAPER_TISSUE_MODES
                else 0.0
            )
            env.physics_settings.triangle_skin_contact_min_volume_ratio = (
                PAPER_SOFT_TISSUE_CONTACT_MIN_VOLUME_RATIO
                if tissue_mode in PAPER_TISSUE_MODES
                else ADAPTIVE_TISSUE_CONTACT_MIN_VOLUME_RATIO
            )
        env.physics_settings.preserve_static_body_poses = True
        if tissue_mode == "paper_pbd":
            print(
                "[super_embodied] paper_pbd fixed constraint baseline: "
                f"k_dist={PAPER_CONSTRAINT_DISTANCE_STIFFNESS:g}, "
                f"k_vol={PAPER_CONSTRAINT_VOLUME_STIFFNESS:g}, "
                f"k_shape={PAPER_CONSTRAINT_SHAPE_STIFFNESS:g}, "
                f"gravity={scene_gravity_m_s2:.1f}m/s^2, "
                f"damping={tissue_velocity_damping:.1f}/s, "
                "constraints=distance+volume+shape_matching, "
                "stiffness_optimization=OFF, depth_residual=OFF"
            )
        elif tissue_mode == "paper_soft":
            print(
                "[super_embodied] paper_soft fixed XPBD baseline: "
                f"E={tissue_young_modulus_pa / 1e3:.2f}kPa, "
                f"nu={tissue_poisson_ratio:.2f}, "
                f"gravity={scene_gravity_m_s2:.1f}m/s^2, "
                f"damping={tissue_velocity_damping:.1f}/s, "
                "material_solve="
                f"{PAPER_SOFT_TISSUE_MATERIAL_ITERATIONS}x"
                f"{PAPER_SOFT_TISSUE_MATERIAL_RELAXATION:g}, "
                "volume_support=OFF, local_no_flip=ON, "
                "depth_residual=OFF, stiffness_optimization=OFF, "
                "shaft_contact=OFF, "
                "jaw_contact=triangle_skin+top_plane, "
                "persistent_grip=ON(bilateral+2_per_jaw_u_t), "
                "grip_min_contact_samples_per_jaw="
                f"{PAPER_SOFT_TISSUE_GRIP_MIN_CONTACT_SAMPLES_PER_JAW}, "
                "grip_sustain="
                f"{PAPER_SOFT_TISSUE_GRIP_ACTIVATION_STEPS} solves, "
                "grip_patch_max_separation="
                f"{PAPER_SOFT_TISSUE_GRIP_MAX_PATCH_SEPARATION_M * 1e3:.1f}mm, "
                "grip_compliance="
                f"{PAPER_SOFT_TISSUE_GRIP_COMPLIANCE_M_PER_N:g}m/N, "
                "grip_transfer="
                f"{PAPER_SOFT_TISSUE_GRIP_TRANSFER_LAYERS} tet layers, "
                "grip_capture_max_penetration="
                f"{PAPER_SOFT_TISSUE_GRIP_MAX_CAPTURE_PENETRATION_M * 1e3:.1f}mm, "
                f"grip_support={PAPER_SOFT_TISSUE_GRIP_SUPPORT_RADIUS_M * 1e3:.1f}mm"
                f"x{PAPER_SOFT_TISSUE_GRIP_SUPPORT_GENERATIONS}, "
                "jaw_friction="
                f"{PAPER_SOFT_TISSUE_JAW_FRICTION_COEFFICIENT:g}, "
                "jaw_query="
                f"{ADAPTIVE_TISSUE_CONTACT_QUERY_DISTANCE_M * 1e3:.1f}mm, "
                "surface_max_correction="
                f"{PAPER_SOFT_TISSUE_SURFACE_ITERATIONS}x"
                f"{PAPER_SOFT_TISSUE_SURFACE_MAX_CORRECTION_M * 1e3:.3f}mm, "
                "top_barrier_max_correction="
                f"{PAPER_SOFT_TISSUE_TOP_BARRIER_MAX_CORRECTION_M * 1e3:.3f}mm, "
                "top_barrier_band="
                f"{PAPER_SOFT_TISSUE_TOP_BARRIER_TIP_ALLOWANCE_M * 1e3:.2f}.."
                f"{PAPER_SOFT_TISSUE_TOP_BARRIER_DISTAL_LENGTH_M * 1e3:.2f}mm, "
                "final_barrier="
                f"{PAPER_SOFT_TISSUE_FINAL_BARRIER_MAX_CORRECTION_M * 1e3:.3f}mm, "
                "pressure_shoulder="
                f"{PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_RADIUS_M * 1e3:.1f}mm@"
                f"up{PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_UPWARD_SCALE:g}/"
                f"out{PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_OUTWARD_SCALE:g}, "
                "left_bias_start="
                f"{PAPER_SOFT_TISSUE_PRESSURE_SHOULDER_BIAS_START_M * 1e3:.1f}mm, "
                "top_barrier_lateral_tol="
                f"{PAPER_SOFT_TISSUE_TOP_BARRIER_LATERAL_TOLERANCE_M * 1e3:.2f}mm, "
                "top_barrier_patch_radius="
                f"{PAPER_SOFT_TISSUE_TOP_BARRIER_CONTACT_PATCH_RADIUS_M * 1e3:.2f}mm, "
                "contact_substep_stride="
                f"{PAPER_SOFT_TISSUE_CONTACT_SUBSTEP_STRIDE}, "
                "contact_spread_layers="
                f"{PAPER_SOFT_TISSUE_CONTACT_SPREAD_LAYERS}, "
                "top_clearance="
                f"{PAPER_SOFT_TISSUE_TOP_CLEARANCE_M * 1e3:.2f}mm, "
                "gaussian_skinning=ON, "
                f"visual_force=PRIMARY(lr={SUPER_VISUAL_FORCE_LR_MEANS:g},"
                f"kp={SUPER_VISUAL_FORCE_KP:g},"
                "caps="
                f"{SUPER_SOFT_VISUAL_FORCE_MAX_GAUSSIAN_N * 1e3:.3f}/"
                f"{SUPER_SOFT_VISUAL_FORCE_MAX_PARTICLE_N * 1e3:.3f}mN,"
                f"total={SUPER_SOFT_VISUAL_FORCE_MAX_TOTAL_N:g}N,"
                "particle_accel<="
                f"{SUPER_SOFT_VISUAL_FORCE_MAX_PARTICLE_ACCELERATION_M_S2:g}m/s^2)"
            )
        else:
            print(
                "[super_embodied] adaptive tissue mechanics: "
                f"E={tissue_young_modulus_pa / 1e3:.2f}kPa, "
                f"nu={tissue_poisson_ratio:.2f}, "
                f"gravity={scene_gravity_m_s2:.1f}m/s^2, "
                f"damping={tissue_velocity_damping:.1f}/s, "
                "contact_stride=1, "
                f"contact_spread_layers={ADAPTIVE_TISSUE_CONTACT_SPREAD_LAYERS}, "
                "top_contact_support="
                f"{ADAPTIVE_TISSUE_TOP_CONTACT_SUPPORT_RADIUS_M * 1e3:.2f}x"
                f"{ADAPTIVE_TISSUE_TOP_CONTACT_SUPPORT_DEPTH_M * 1e3:.2f}mm, "
                f"contact_margin={ADAPTIVE_TISSUE_CONTACT_MARGIN_M * 1e3:.2f}mm, "
                "contact_query="
                f"{ADAPTIVE_TISSUE_CONTACT_QUERY_DISTANCE_M * 1e3:.1f}mm, "
                "contact_max_correction="
                f"{ADAPTIVE_TISSUE_CONTACT_MAX_CORRECTION_M * 1e3:.2f}mm, "
                "contact_min_volume_ratio="
                f"{ADAPTIVE_TISSUE_CONTACT_MIN_VOLUME_RATIO:.4g}, "
                "material_min_volume_ratio="
                f"{tissue_material_min_volume_ratio:.4g}, "
                "kinematic_pose_retraction="
                f"{'ON' if ADAPTIVE_TISSUE_ENABLE_KINEMATIC_GAP_GUARD else 'OFF'}"
            )
        env.sim.visual_forces.configure_body_participation(
            gradient_body_ids=[],
            physics_force_body_ids=[],
        )
        env.sim.visual_forces.configure_gaussian_participation(
            env.sim.gaussian_model.soft_gaussian_ids
        )
        visual_settings.lr_quats = 0.0
        visual_settings.iterations = 1
        visual_settings.enable_soft_particle_forces = True
        visual_settings.soft_max_gaussian_force = (
            SUPER_SOFT_VISUAL_FORCE_MAX_GAUSSIAN_N
        )
        visual_settings.soft_max_particle_force = (
            SUPER_SOFT_VISUAL_FORCE_MAX_PARTICLE_N
        )
        visual_settings.soft_max_total_force = (
            SUPER_SOFT_VISUAL_FORCE_MAX_TOTAL_N
        )
        visual_settings.soft_max_particle_acceleration = (
            SUPER_SOFT_VISUAL_FORCE_MAX_PARTICLE_ACCELERATION_M_S2
        )
        visual_settings.soft_force_spread_layers = (
            SUPER_SOFT_VISUAL_FORCE_SPREAD_LAYERS
        )
    else:
        env.sim.visual_forces.configure_body_participation(
            gradient_body_ids=tissue_body_ids,
            physics_force_body_ids=tissue_body_ids,
        )
        visual_settings.lr_quats = SUPER_VISUAL_FORCE_LR_QUATS
        visual_settings.enable_soft_particle_forces = False
    apply_psm_lnd_pose(env, 0)
    env.sim.configure_kinematic_body_interpolation(
        env.super_psm_lnd_body_ids[0]  # type: ignore[attr-defined]
    )
    env.stash_state()
    return env
