#!/usr/bin/env python3

"""Propagate body/image-left-jaw/image-right-jaw masks over SUPER grasp5.

The three object identities are initialized from the registered strict-LND CAD
projection on frame 0.  Validated manual masks are injected at their keyframes,
and sparse LND-projected conditioning masks keep the three SurgicalSAM2
memories from exchanging identities over the long sequence.  Low confidence is
recorded explicitly for the downstream pose refiner; it is never hidden by
fabricating a high-confidence observation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
PAPER_REPO = Path("/Media_HDD/jwshan/wad/online_dvrk_tracking")
TRACK_ROOT = REPO / "data/super/psm_tracking"
sys.path[:0] = [
    str(REPO),
    str(REPO / "scripts"),
    str(PAPER_REPO),
    str(PAPER_REPO / "SurgicalSAM2"),
]

from refine_super_psm_part_keyframes import (  # noqa: E402
    PART_COLORS,
    PART_NAMES,
    load_part_observation,
    paper_lnd_prior,
)
from super_psm_tracking_common import (  # noqa: E402
    TrackingInputs,
    _paper_component_transforms,
    ctr_to_matrix,
)


OBJ_ID_BY_NAME = {name: index for index, name in enumerate(PART_NAMES)}
MANUAL_ANCHORS = (141, 142, 160, 320)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-object SurgicalSAM2 propagation on SUPER grasp5."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=REPO
        / "data/super/grasp5_offline_demo/videos/stereo_left.mp4",
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=TRACK_ROOT / "tracking_states_paper_exact.npz",
    )
    parser.add_argument(
        "--manual-masks-dir",
        type=Path,
        default=TRACK_ROOT / "part_segmentation_experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRACK_ROOT / "part_masks_full_sequence",
    )
    parser.add_argument("--reprompt-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--diagnostic-frames",
        default="0,141,142,160,320,480,640,800,960,1120,1280,1440",
    )
    return parser.parse_args()


def paper_prior_sequence(
    inputs: TrackingInputs,
    exact: np.lib.npyio.NpzFile,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    ctr = []
    joints = []
    for frame_index in range(frame_count):
        frame_ctr, frame_joints = paper_lnd_prior(
            inputs,
            exact["pure_ctr"],
            exact["pure_joints"],
            frame_index,
        )
        ctr.append(frame_ctr)
        joints.append(frame_joints)
    return np.asarray(ctr, dtype=np.float32), np.asarray(joints, dtype=np.float32)


def project_prior_landmarks(
    inputs: TrackingInputs,
    ctr: np.ndarray,
    joints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    K = inputs.K_half
    tips = np.empty((len(ctr), 2, 2), dtype=np.float32)
    pivots = np.empty((len(ctr), 2), dtype=np.float32)
    local_points = (
        np.asarray([0.0, 0.0004, 0.0096, 1.0]),
        np.asarray([0.0, -0.0004, 0.0096, 1.0]),
    )
    for frame_index, (frame_ctr, frame_joints) in enumerate(
        zip(ctr, joints, strict=True)
    ):
        camera = ctr_to_matrix(frame_ctr)
        components = _paper_component_transforms(frame_joints)
        for tip_index, (component_index, point) in enumerate(
            zip((3, 4), local_points, strict=True)
        ):
            camera_point = camera @ components[component_index] @ point
            tips[frame_index, tip_index] = (
                K @ (camera_point[:3] / camera_point[2])
            )[:2]
        pivot_camera = camera @ components[2] @ np.asarray([0, 0, 0, 1])
        pivots[frame_index] = (
            K @ (pivot_camera[:3] / pivot_camera[2])
        )[:2]
    return tips, pivots


def load_paper_meshes() -> tuple[
    tuple[np.ndarray, ...], tuple[np.ndarray, ...]
]:
    from pytorch3d.io import load_ply

    mesh_dir = PAPER_REPO / "urdfs/dVRK/meshes"
    paths = (
        mesh_dir / "low_res_shaft_multi_cylinder.ply",
        mesh_dir / "low_res_logo_low_res_1.ply",
        mesh_dir / "low_res_jawright_lowres.ply",
        mesh_dir / "low_res_jawleft_lowres.ply",
    )
    vertices = []
    faces = []
    for path in paths:
        mesh_vertices, mesh_faces = load_ply(str(path))
        vertices.append(mesh_vertices.numpy().astype(np.float64))
        faces.append(mesh_faces.numpy().astype(np.int32))
    return tuple(vertices), tuple(faces)


def rasterize_mesh_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    transform: np.ndarray,
    K: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    camera_vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
    projected = np.empty((len(vertices), 2), dtype=np.float64)
    projected[:, 0] = (
        K[0, 0] * camera_vertices[:, 0] / camera_vertices[:, 2] + K[0, 2]
    )
    projected[:, 1] = (
        K[1, 1] * camera_vertices[:, 1] / camera_vertices[:, 2] + K[1, 2]
    )
    valid_faces = faces[np.all(camera_vertices[faces, 2] > 1e-5, axis=1)]
    polygons = np.rint(projected[valid_faces]).astype(np.int32)
    polygons = np.clip(polygons, -10000, 10000)
    mask = np.zeros(shape, dtype=np.uint8)
    if len(polygons):
        cv2.fillPoly(mask, list(polygons), 1)
    return mask.astype(bool)


def render_paper_parts_cpu(
    ctr: np.ndarray,
    joints: np.ndarray,
    vertices: tuple[np.ndarray, ...],
    faces: tuple[np.ndarray, ...],
    K: np.ndarray,
    shape: tuple[int, int] = (540, 960),
) -> dict[str, np.ndarray]:
    camera = ctr_to_matrix(ctr)
    components = _paper_component_transforms(joints)
    component_by_mesh = (0, 1, 3, 4)
    mesh_masks = [
        rasterize_mesh_silhouette(
            vertices[mesh_index],
            faces[mesh_index],
            camera @ components[component_by_mesh[mesh_index]],
            K,
            shape,
        )
        for mesh_index in range(4)
    ]
    return {
        "body": mesh_masks[0] | mesh_masks[1],
        "jaw_left": mesh_masks[2],
        "jaw_right": mesh_masks[3],
    }


def bbox_from_mask(
    mask: np.ndarray,
    *,
    margin: int,
) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot create a box from an empty mask")
    height, width = mask.shape
    return np.asarray(
        [
            max(0, int(xs.min()) - margin),
            max(0, int(ys.min()) - margin),
            min(width - 1, int(xs.max()) + margin),
            min(height - 1, int(ys.max()) + margin),
        ],
        dtype=np.float32,
    )


def interior_points(mask: np.ndarray, count: int) -> np.ndarray:
    binary = mask.astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    points = []
    working = distance.copy()
    for _ in range(count):
        _, maximum, _, location = cv2.minMaxLoc(working)
        if maximum <= 0:
            break
        points.append(location)
        cv2.circle(working, location, 18, 0.0, -1)
    if not points:
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise ValueError("Cannot select a point from an empty mask")
        points.append((int(np.median(xs)), int(np.median(ys))))
    return np.asarray(points, dtype=np.float32)


def guided_prompts(
    rendered: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    binary = {name: rendered[name] >= 0.25 for name in PART_NAMES}
    centers = {
        name: interior_points(mask, 3 if name == "body" else 2)
        for name, mask in binary.items()
    }
    output = {}
    for name in PART_NAMES:
        positive = centers[name]
        negative = np.concatenate(
            [centers[other][:1] for other in PART_NAMES if other != name],
            axis=0,
        )
        points = np.concatenate((positive, negative), axis=0)
        labels = np.asarray(
            [1] * len(positive) + [0] * len(negative), dtype=np.int32
        )
        box = bbox_from_mask(
            binary[name], margin=25 if name == "body" else 35
        )
        output[name] = (points, labels, box)
    return output


def guided_jaw_tip(
    mask: np.ndarray,
    predicted_tip: np.ndarray,
    predicted_pivot: np.ndarray,
    *,
    search_radius: float = 45.0,
) -> tuple[np.ndarray, bool]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return predicted_tip.astype(np.float32), False
    points = np.concatenate(contours, axis=0).reshape(-1, 2).astype(np.float32)
    tip_distance = np.linalg.norm(points - predicted_tip, axis=1)
    pivot_distance = np.linalg.norm(points - predicted_pivot, axis=1)
    candidate = tip_distance <= search_radius
    if not np.any(candidate):
        return predicted_tip.astype(np.float32), False
    candidate_ids = np.flatnonzero(candidate)
    score = pivot_distance[candidate_ids] - 0.2 * tip_distance[candidate_ids]
    return points[candidate_ids[np.argmax(score)]], True


def point_distance_to_mask(mask: np.ndarray, point: np.ndarray) -> float:
    x, y = np.rint(point).astype(int)
    height, width = mask.shape
    if 0 <= x < width and 0 <= y < height and mask[y, x]:
        return 0.0
    distance = cv2.distanceTransform(
        (~mask).astype(np.uint8), cv2.DIST_L2, 3
    )
    if not (0 <= x < width and 0 <= y < height):
        return float(max(height, width))
    return float(distance[y, x])


def mask_confidences(
    masks: dict[str, np.ndarray],
    previous_masks: dict[str, np.ndarray] | None,
    prior_tips: np.ndarray,
    prior_pivot: np.ndarray,
    *,
    manual: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if manual:
        confidence = np.ones(3, dtype=np.float32)
    else:
        confidence = np.empty(3, dtype=np.float32)
        for part_index, name in enumerate(PART_NAMES):
            mask = masks[name]
            area = int(mask.sum())
            if name == "body":
                area_score = np.clip(
                    min(area / 7000.0, 70000.0 / max(area, 1)), 0.0, 1.0
                )
                location_distance = point_distance_to_mask(mask, prior_pivot)
                location_score = math_exp_decay(location_distance, 35.0)
            else:
                area_score = np.clip(
                    min(area / 150.0, 12000.0 / max(area, 1)), 0.0, 1.0
                )
                tip_index = 0 if name == "jaw_left" else 1
                location_distance = point_distance_to_mask(
                    mask, prior_tips[tip_index]
                )
                location_score = math_exp_decay(location_distance, 25.0)
            temporal_score = 1.0
            if previous_masks is not None:
                previous = previous_masks[name]
                union = np.count_nonzero(mask | previous)
                temporal_iou = (
                    np.count_nonzero(mask & previous) / union if union else 0.0
                )
                temporal_score = np.clip(0.35 + temporal_iou, 0.0, 1.0)
            confidence[part_index] = float(
                area_score * location_score * temporal_score
            )
    tips = np.empty((2, 2), dtype=np.float32)
    tip_valid = np.empty(2, dtype=bool)
    for tip_index, name in enumerate(("jaw_left", "jaw_right")):
        tips[tip_index], tip_valid[tip_index] = guided_jaw_tip(
            masks[name], prior_tips[tip_index], prior_pivot
        )
        if not tip_valid[tip_index]:
            confidence[tip_index + 1] *= 0.25
    overlap = np.asarray(
        [
            np.count_nonzero(masks["body"] & masks["jaw_left"]),
            np.count_nonzero(masks["body"] & masks["jaw_right"]),
            np.count_nonzero(masks["jaw_left"] & masks["jaw_right"]),
        ],
        dtype=np.int32,
    )
    jaw_union = np.count_nonzero(masks["jaw_left"] | masks["jaw_right"])
    if jaw_union and overlap[2] / jaw_union > 0.35:
        confidence[1:] *= 0.5
    return confidence, tips, overlap


def math_exp_decay(value: float, scale: float) -> float:
    return float(np.exp(-max(value, 0.0) / scale))


def logits_to_masks(
    obj_ids: list[int], logits: torch.Tensor
) -> dict[str, np.ndarray]:
    logits_np = logits.detach().float().cpu().numpy()
    if logits_np.ndim == 4 and logits_np.shape[1] == 1:
        logits_np = logits_np[:, 0]
    if logits_np.ndim != 3:
        raise ValueError(f"Unexpected SAM logits shape {logits_np.shape}")
    by_id = {
        int(obj_id): logits_np[index] > 0
        for index, obj_id in enumerate(obj_ids)
    }
    return {
        name: by_id[OBJ_ID_BY_NAME[name]].astype(bool) for name in PART_NAMES
    }


def append_current_image_for_prompt(predictor: object, frame: np.ndarray) -> None:
    prepared, _, _ = predictor.perpare_data(
        frame, image_size=predictor.image_size
    )
    images = predictor.condition_state["images"]
    target_index = predictor.condition_state["num_frames"] - 1
    while len(images) <= target_index:
        images.append(prepared)


def add_masks_on_current_frame(
    predictor: object,
    masks: dict[str, np.ndarray],
) -> tuple[list[int], torch.Tensor]:
    frame_idx = predictor.condition_state["num_frames"] - 1
    output = None
    for name in PART_NAMES:
        output = predictor.add_new_mask(
            frame_idx,
            OBJ_ID_BY_NAME[name],
            masks[name],
        )
    assert output is not None
    _, obj_ids, logits = output
    return obj_ids, logits


def add_prompts_on_current_frame(
    predictor: object,
    prompts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[list[int], torch.Tensor]:
    frame_idx = predictor.condition_state["num_frames"] - 1
    output = None
    for name in PART_NAMES:
        points, labels, box = prompts[name]
        output = predictor.add_new_prompt(
            frame_idx,
            OBJ_ID_BY_NAME[name],
            points=points,
            labels=labels,
            bbox=box,
            clear_old_points=True,
        )
    assert output is not None
    _, obj_ids, logits = output
    return obj_ids, logits


def save_checkpoint(
    path: Path,
    *,
    packed_masks: list[np.ndarray],
    mask_shape: tuple[int, int],
    confidences: list[np.ndarray],
    tips: list[np.ndarray],
    areas: list[np.ndarray],
    overlaps: list[np.ndarray],
    prompt_source: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        part_names=np.asarray(PART_NAMES),
        masks_packbits=np.stack(packed_masks),
        mask_shape=np.asarray(mask_shape, dtype=np.int32),
        confidence=np.asarray(confidences, dtype=np.float32),
        jaw_tips_xy=np.asarray(tips, dtype=np.float32),
        areas_px=np.asarray(areas, dtype=np.int32),
        pairwise_overlap_px=np.asarray(overlaps, dtype=np.int32),
        prompt_source=np.asarray(prompt_source, dtype=np.int8),
    )


def draw_diagnostic(
    frame: np.ndarray,
    masks: dict[str, np.ndarray],
    tips: np.ndarray,
    confidence: np.ndarray,
    frame_index: int,
    source: int,
) -> np.ndarray:
    output = frame.copy()
    for name in PART_NAMES:
        mask = masks[name]
        tint = np.zeros_like(output)
        tint[mask] = PART_COLORS[name]
        output = cv2.addWeighted(output, 1.0, tint, 0.4, 0.0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, PART_COLORS[name], 2)
    for point in tips:
        cv2.circle(output, tuple(np.rint(point).astype(int)), 6, (0, 255, 255), -1)
    cv2.rectangle(output, (0, 0), (output.shape[1], 62), (0, 0, 0), -1)
    source_name = {0: "track", 1: "LND mask", 2: "manual mask"}[source]
    cv2.putText(
        output,
        f"frame {frame_index}: {source_name}; confidence={np.round(confidence, 2).tolist()}",
        (12, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.video.exists() or not args.states.exists():
        raise FileNotFoundError(args.video if not args.video.exists() else args.states)
    diagnostic_frames = {
        int(value) for value in args.diagnostic_frames.split(",") if value
    }

    inputs = TrackingInputs.load()
    exact = np.load(args.states)
    cap = cv2.VideoCapture(str(args.video))
    video_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = min(video_count, len(inputs.video_timestamps))
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)
    prior_ctr, prior_joints = paper_prior_sequence(
        inputs, exact, frame_count
    )
    prior_tips, prior_pivots = project_prior_landmarks(
        inputs, prior_ctr, prior_joints
    )

    from sam2.build_sam import build_sam2_camera_predictor

    mesh_vertices, mesh_faces = load_paper_meshes()
    guidance_frames = {
        frame_index
        for frame_index in range(frame_count)
        if frame_index == 0
        or (
            args.reprompt_every > 0
            and frame_index % args.reprompt_every == 0
            and frame_index not in MANUAL_ANCHORS
        )
    }
    initial_masks: dict[str, np.ndarray] | None = None
    guidance_masks: dict[int, dict[str, np.ndarray]] = {}
    for frame_index in sorted(guidance_frames):
        rendered_prior = render_paper_parts_cpu(
            prior_ctr[frame_index],
            prior_joints[frame_index],
            mesh_vertices,
            mesh_faces,
            inputs.K_half,
        )
        if frame_index == 0:
            initial_masks = {
                name: rendered_prior[name] >= 0.25 for name in PART_NAMES
            }
        else:
            guidance_masks[frame_index] = {
                name: rendered_prior[name] >= 0.25 for name in PART_NAMES
            }
    assert initial_masks is not None

    predictor = build_sam2_camera_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(
            PAPER_REPO
            / "SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth"
        ),
        vos_optimized=False,
    )

    packed_masks: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    tip_history: list[np.ndarray] = []
    areas: list[np.ndarray] = []
    overlaps: list[np.ndarray] = []
    prompt_sources: list[int] = []
    diagnostics = []
    previous_masks: dict[str, np.ndarray] | None = None
    partial_path = args.output_dir / "part_masks_partial.npz"
    started = time.perf_counter()
    mask_shape: tuple[int, int] | None = None

    for frame_index in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index}")
        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        manual = frame_index in MANUAL_ANCHORS
        auto_prompt = (
            frame_index == 0
            or (
                args.reprompt_every > 0
                and frame_index % args.reprompt_every == 0
            )
        )
        source = 0
        manual_tips: dict[str, np.ndarray] | None = None
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            if frame_index == 0:
                predictor.load_first_frame(frame)
                obj_ids, logits = add_masks_on_current_frame(
                    predictor, initial_masks
                )
                masks = logits_to_masks(obj_ids, logits)
                source = 1
            else:
                obj_ids, logits = predictor.track(frame)
                masks = logits_to_masks(obj_ids, logits)
                if manual or auto_prompt:
                    append_current_image_for_prompt(predictor, frame)
                    predictor.condition_state["tracking_has_started"] = False
                    if manual:
                        manual_masks, manual_tips = load_part_observation(
                            args.manual_masks_dir, frame_index
                        )
                        obj_ids, logits = add_masks_on_current_frame(
                            predictor, manual_masks
                        )
                        masks = manual_masks
                        source = 2
                    else:
                        obj_ids, logits = add_masks_on_current_frame(
                            predictor, guidance_masks[frame_index]
                        )
                        masks = guidance_masks[frame_index]
                        source = 1

        mask_shape = masks["body"].shape
        confidence, jaw_tips, frame_overlap = mask_confidences(
            masks,
            previous_masks,
            prior_tips[frame_index],
            prior_pivots[frame_index],
            manual=manual,
        )
        if manual_tips is not None:
            jaw_tips = np.stack(
                [manual_tips["jaw_left"], manual_tips["jaw_right"]]
            ).astype(np.float32)
        if source == 1:
            # A CAD conditioning mask preserves object identity but is not an
            # image-derived observation.  Cap its downstream influence.
            confidence = np.minimum(
                confidence,
                np.asarray([0.45, 0.35, 0.35], dtype=np.float32),
            )
        packed_masks.append(
            np.stack(
                [
                    np.packbits(masks[name].reshape(-1))
                    for name in PART_NAMES
                ]
            )
        )
        confidences.append(confidence)
        tip_history.append(jaw_tips)
        areas.append(
            np.asarray([masks[name].sum() for name in PART_NAMES], dtype=np.int32)
        )
        overlaps.append(frame_overlap)
        prompt_sources.append(source)
        previous_masks = {name: masks[name].copy() for name in PART_NAMES}

        if frame_index in diagnostic_frames:
            diagnostic = draw_diagnostic(
                frame,
                masks,
                jaw_tips,
                confidence,
                frame_index,
                source,
            )
            path = args.output_dir / f"frame{frame_index:06d}_parts.png"
            cv2.imwrite(str(path), diagnostic)
            diagnostics.append(cv2.resize(diagnostic, (960, 540)))
        elapsed = time.perf_counter() - started
        print(
            f"part_masks frame={frame_index:04d}/{frame_count - 1:04d} "
            f"source={source} confidence={np.round(confidence, 2).tolist()} "
            f"areas={areas[-1].tolist()} elapsed={elapsed:.1f}s"
        )
        if (
            (frame_index + 1) % args.checkpoint_every == 0
            or frame_index + 1 == frame_count
        ):
            assert mask_shape is not None
            save_checkpoint(
                partial_path,
                packed_masks=packed_masks,
                mask_shape=mask_shape,
                confidences=confidences,
                tips=tip_history,
                areas=areas,
                overlaps=overlaps,
                prompt_source=prompt_sources,
            )
    cap.release()

    full_run = frame_count == video_count == len(inputs.video_timestamps)
    output_path = args.output_dir / (
        "part_masks_full.npz"
        if full_run
        else f"part_masks_preview_{frame_count}.npz"
    )
    partial_path.replace(output_path)
    confidence_array = np.asarray(confidences)
    report = {
        "frame_count": frame_count,
        "full_run": full_run,
        "part_names": list(PART_NAMES),
        "manual_anchors": [
            index for index in MANUAL_ANCHORS if index < frame_count
        ],
        "reprompt_every": args.reprompt_every,
        "prompt_source_counts": {
            "track": int(np.count_nonzero(np.asarray(prompt_sources) == 0)),
            "lnd_mask": int(np.count_nonzero(np.asarray(prompt_sources) == 1)),
            "manual_mask": int(np.count_nonzero(np.asarray(prompt_sources) == 2)),
        },
        "confidence_p05_p50_p95": {
            name: np.percentile(confidence_array[:, part_index], [5, 50, 95]).tolist()
            for part_index, name in enumerate(PART_NAMES)
        },
        "low_confidence_below_0_35": {
            name: int(np.count_nonzero(confidence_array[:, part_index] < 0.35))
            for part_index, name in enumerate(PART_NAMES)
        },
        "elapsed_seconds": time.perf_counter() - started,
        "asset": str(output_path),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if diagnostics:
        cv2.imwrite(
            str(args.output_dir / "contact_sheet.png"), np.vstack(diagnostics)
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
