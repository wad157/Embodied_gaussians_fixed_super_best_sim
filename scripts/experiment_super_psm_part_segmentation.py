#!/usr/bin/env python3

"""Diagnose PSM tracking with separate body/left-jaw/right-jaw SAM masks.

This is a key-frame experiment, not part of the paper-exact estimator.  Each
part receives its own positive points, negative points on the other parts, and
a tight box.  The smallest SAM multimask candidate satisfying the prompts is
kept.  The purpose is to test whether part-aware observations remove the false
shaft-tip detection seen at frame 142 and recover the omitted jaw at frame 320.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL.Image import Image


REPO = Path(__file__).resolve().parents[1]
PAPER_REPO = Path("/Media_HDD/jwshan/wad/online_dvrk_tracking")
TRACK_ROOT = REPO / "data/super/psm_tracking"
sys.path[:0] = [
    str(PAPER_REPO / "SurgicalSAM2"),
    str(PAPER_REPO),
]

from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402


@dataclass(frozen=True)
class PartPrompt:
    positive: tuple[tuple[float, float], ...]
    negative: tuple[tuple[float, float], ...]
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class FramePrompts:
    pivot: tuple[float, float]
    body: PartPrompt
    jaw_left: PartPrompt
    jaw_right: PartPrompt


# Coordinates refer to the 960x540 frames used by the paper tracker.  These
# hand-picked key-frame prompts deliberately provide an upper-bound test of
# separate part masks before adding automatic optical-flow prompt propagation.
FRAME_PROMPTS: dict[int, FramePrompts] = {
    141: FramePrompts(
        pivot=(430, 224),
        body=PartPrompt(
            positive=((430, 205), (480, 180), (535, 125), (620, 55)),
            negative=((360, 229), (458, 282)),
            box=(400, 0, 900, 240),
        ),
        jaw_left=PartPrompt(
            positive=((360, 229), (390, 227), (418, 224)),
            negative=((452, 260), (480, 180)),
            box=(330, 200, 440, 255),
        ),
        jaw_right=PartPrompt(
            positive=((458, 282), (452, 260), (444, 240)),
            negative=((390, 227), (480, 180)),
            box=(420, 215, 485, 310),
        ),
    ),
    142: FramePrompts(
        pivot=(430, 224),
        body=PartPrompt(
            positive=((430, 205), (480, 180), (550, 110), (650, 45)),
            negative=((360, 226), (458, 282)),
            box=(400, 0, 900, 240),
        ),
        jaw_left=PartPrompt(
            positive=((360, 226), (390, 226), (420, 225)),
            negative=((450, 250), (480, 180)),
            box=(330, 200, 440, 255),
        ),
        jaw_right=PartPrompt(
            positive=((458, 282), (452, 260), (445, 240)),
            negative=((390, 226), (480, 180)),
            box=(420, 215, 485, 310),
        ),
    ),
    160: FramePrompts(
        pivot=(423, 232),
        body=PartPrompt(
            positive=((425, 215), (470, 190), (525, 140), (620, 60)),
            negative=((351, 238), (450, 286)),
            box=(395, 0, 900, 245),
        ),
        jaw_left=PartPrompt(
            positive=((351, 238), (382, 235), (410, 231)),
            negative=((446, 260), (470, 190)),
            box=(325, 210, 435, 265),
        ),
        jaw_right=PartPrompt(
            positive=((450, 286), (446, 262), (438, 242)),
            negative=((382, 235), (470, 190)),
            box=(412, 218, 482, 312),
        ),
    ),
    320: FramePrompts(
        pivot=(397, 306),
        body=PartPrompt(
            positive=((410, 292), (455, 260), (505, 220), (565, 165)),
            negative=((318, 329), (408, 380)),
            box=(375, 105, 900, 325),
        ),
        jaw_left=PartPrompt(
            positive=((318, 329), (350, 321), (382, 310)),
            negative=((405, 350), (440, 275)),
            box=(295, 295, 408, 352),
        ),
        jaw_right=PartPrompt(
            positive=((408, 380), (405, 352), (401, 323)),
            negative=((350, 321), (440, 275)),
            box=(378, 296, 438, 408),
        ),
    ),
}


class Torch26SAM2ImagePredictor(SAM2ImagePredictor):
    """Use reshape and actual backbone sizes with the SurgicalSAM2 checkpoint.

    SurgicalSAM2's copied single-image wrapper assumes 1024px SAM2 features
    and calls view() after permute().  Its endoscopic checkpoint is 512px and
    PyTorch 2.6 rejects that non-contiguous view.  The camera/video predictor
    already handles the checkpoint; this local compatibility wrapper only
    fixes the single-image experiment interface.
    """

    @torch.no_grad()
    def set_image(self, image: np.ndarray | Image) -> None:
        self.reset_predictor()
        if isinstance(image, np.ndarray):
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            width, height = image.size
            self._orig_hw = [(height, width)]
        else:
            raise NotImplementedError("Unsupported image type")

        input_image = self._transforms(image)[None, ...].to(self.device)
        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feature_sizes = []
        for feature in vision_feats:
            side = int(round(math.sqrt(feature.shape[0])))
            if side * side != feature.shape[0]:
                raise ValueError(f"Non-square SAM feature map: {feature.shape}")
            feature_sizes.append((side, side))
        feats = [
            feature.permute(1, 2, 0).reshape(1, -1, *feature_size)
            for feature, feature_size in zip(
                vision_feats[::-1], feature_sizes[::-1], strict=True
            )
        ][::-1]
        self._features = {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
        }
        self._is_image_set = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="141,142,160,320")
    parser.add_argument(
        "--states",
        type=Path,
        default=TRACK_ROOT / "tracking_states_paper_exact.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRACK_ROOT / "part_segmentation_experiment",
    )
    return parser.parse_args()


def unpack_original_mask(states: np.lib.npyio.NpzFile, frame_index: int) -> np.ndarray:
    shape = tuple(states["mask_shape"].tolist())
    return np.unpackbits(states["masks_packbits"][frame_index])[
        : np.prod(shape)
    ].reshape(shape).astype(bool)


def run_part_prediction(
    predictor: SAM2ImagePredictor,
    prompt: PartPrompt,
) -> tuple[np.ndarray, int, float, list[int], list[float]]:
    points = np.asarray(prompt.positive + prompt.negative, dtype=np.float32)
    labels = np.asarray(
        [1] * len(prompt.positive) + [0] * len(prompt.negative),
        dtype=np.int64,
    )
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=np.asarray(prompt.box, dtype=np.float32),
        multimask_output=True,
    )
    masks_bool = np.asarray(masks) > 0
    positive_xy = np.rint(points[labels == 1]).astype(int)
    negative_xy = np.rint(points[labels == 0]).astype(int)
    ranking = []
    for candidate_index, mask in enumerate(masks_bool):
        positive_misses = sum(not mask[y, x] for x, y in positive_xy)
        negative_hits = sum(mask[y, x] for x, y in negative_xy)
        ranking.append(
            (
                positive_misses + negative_hits,
                int(mask.sum()),
                -float(scores[candidate_index]),
                candidate_index,
            )
        )
    selected_index = min(ranking)[-1]
    return (
        masks_bool[selected_index],
        selected_index,
        float(scores[selected_index]),
        [int(mask.sum()) for mask in masks_bool],
        [float(score) for score in scores],
    )


def jaw_tip(mask: np.ndarray, pivot: tuple[float, float]) -> tuple[int, int]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("Jaw mask has no contour")
    points = max(contours, key=cv2.contourArea).reshape(-1, 2)
    pivot_xy = np.asarray(pivot, dtype=np.float32)
    return tuple(points[np.argmax(np.linalg.norm(points - pivot_xy, axis=1))])


def tint_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    tint = np.zeros_like(image)
    tint[mask] = color
    cv2.addWeighted(image, 1.0, tint, 0.45, 0.0, dst=image)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, color, 2)


def add_title(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    frame_indices = [int(value) for value in args.frames.split(",") if value]
    unsupported = sorted(set(frame_indices) - set(FRAME_PROMPTS))
    if unsupported:
        raise ValueError(f"No prompts defined for frames {unsupported}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(
            PAPER_REPO
            / "SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth"
        ),
    )
    predictor = Torch26SAM2ImagePredictor(model)
    states = np.load(args.states)
    metrics: dict[str, object] = {}
    comparison_images = []
    colors = {
        "body": (0, 180, 255),
        "jaw_left": (255, 0, 255),
        "jaw_right": (0, 0, 255),
    }

    for frame_index in frame_indices:
        native_path = (
            REPO
            / f"data/super/grasp5_native/rgb/{frame_index:06d}-left.png"
        )
        native = cv2.imread(str(native_path))
        if native is None:
            raise FileNotFoundError(native_path)
        frame = cv2.resize(native, (960, 540), interpolation=cv2.INTER_AREA)
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_prompts = FRAME_PROMPTS[frame_index]
        original_mask = unpack_original_mask(states, frame_index)

        masks: dict[str, np.ndarray] = {}
        selections: dict[str, object] = {}
        for name in ("body", "jaw_left", "jaw_right"):
            prediction = run_part_prediction(
                predictor, getattr(frame_prompts, name)
            )
            mask, selected, score, areas, scores = prediction
            masks[name] = mask
            selections[name] = {
                "selected_candidate": selected,
                "predicted_iou": score,
                "candidate_areas_px": areas,
                "candidate_scores": scores,
                "selected_area_px": int(mask.sum()),
                "outside_original_union_px": int((mask & ~original_mask).sum()),
            }

        left_tip = jaw_tip(masks["jaw_left"], frame_prompts.pivot)
        right_tip = jaw_tip(masks["jaw_right"], frame_prompts.pivot)
        overlap = (
            (masks["body"] & masks["jaw_left"])
            | (masks["body"] & masks["jaw_right"])
            | (masks["jaw_left"] & masks["jaw_right"])
        )
        selections["jaw_left"]["tip_xy"] = [int(value) for value in left_tip]
        selections["jaw_right"]["tip_xy"] = [int(value) for value in right_tip]
        metrics[str(frame_index)] = {
            "original_union_area_px": int(original_mask.sum()),
            "pairwise_overlap_area_px": int(overlap.sum()),
            "parts": selections,
        }

        raw_panel = frame.copy()
        add_title(raw_panel, f"frame {frame_index}: RGB")
        union_panel = frame.copy()
        tint_mask(union_panel, original_mask, (0, 255, 0))
        add_title(union_panel, "paper original: one union mask")
        separated_panel = frame.copy()
        for name in ("body", "jaw_left", "jaw_right"):
            tint_mask(separated_panel, masks[name], colors[name])
        for name, tip in (("L", left_tip), ("R", right_tip)):
            cv2.circle(separated_panel, tip, 6, (0, 255, 255), -1)
            cv2.putText(
                separated_panel,
                name,
                (tip[0] + 7, tip[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        add_title(
            separated_panel,
            "separate: body=orange, left=magenta, right=red, tips=yellow",
        )
        comparison = np.hstack((raw_panel, union_panel, separated_panel))
        comparison_path = args.output_dir / f"frame{frame_index:06d}_parts.png"
        cv2.imwrite(str(comparison_path), comparison)
        comparison_images.append(comparison)
        np.savez_compressed(
            args.output_dir / f"frame{frame_index:06d}_masks.npz",
            body=masks["body"],
            jaw_left=masks["jaw_left"],
            jaw_right=masks["jaw_right"],
            original_union=original_mask,
            jaw_left_tip_xy=np.asarray(left_tip, dtype=np.int32),
            jaw_right_tip_xy=np.asarray(right_tip, dtype=np.int32),
        )
        print(comparison_path)

    cv2.imwrite(
        str(args.output_dir / "contact_sheet.png"),
        np.vstack(comparison_images),
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
