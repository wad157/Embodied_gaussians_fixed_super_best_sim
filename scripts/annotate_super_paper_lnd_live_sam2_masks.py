#!/usr/bin/env python3
"""Live SurgicalSAM2 annotation of shaft/distal masks at stereo anchors.

Each left/right click immediately updates a SAM2 mask.  The user confirms the
mask, not a dense point set.  Shaft and distal-tool masks are collected as
separate objects for each raw strict-stereo prompt anchor.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from experiment_super_psm_part_segmentation import (
    PAPER_REPO,
    Torch26SAM2ImagePredictor,
    build_sam2,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_multianchor_v2"
)
CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_s.yaml"
CHECKPOINT_NAME = "sam2.1_hiera_s_endo18.pth"
PART_COLORS = {
    "tips": (80, 255, 80),
    "distal": (30, 190, 255),
    "shaft": (255, 120, 20),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively create visible SAM2 masks after every positive or "
            "negative click for all stereo prompt anchors."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--paper-repo", type=Path, default=PAPER_REPO)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PAPER_REPO / f"SurgicalSAM2/checkpoints/{CHECKPOINT_NAME}",
    )
    parser.add_argument("--config", default=CONFIG_NAME)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--cuda-device", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DEFAULT_ROOT / "annotations/live_sam2_multianchor_v4"
        ),
    )
    parser.add_argument(
        "--seed-annotation-dir",
        type=Path,
        default=None,
        help=(
            "Reuse completed slot-matched tips and masks from an older "
            "annotation directory; only newly inserted anchors are shown."
        ),
    )
    parser.add_argument("--window-width", type=int, default=1100)
    parser.add_argument("--window-height", type=int, default=720)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def tint_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.48,
) -> np.ndarray:
    output = image.copy()
    overlay = image.copy()
    overlay[mask] = color
    output[mask] = cv2.addWeighted(
        image[mask],
        1.0 - alpha,
        overlay[mask],
        alpha,
        0.0,
    )
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)
    return output


class LiveMaskAnnotator:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        manifest: dict[str, Any],
        predictor: Torch26SAM2ImagePredictor,
    ) -> None:
        self.args = args
        self.root = args.root
        self.manifest = manifest
        self.predictor = predictor
        self.factor = args.downsample_factor
        source_width, source_height = manifest["image_size_wh"]
        if source_width % self.factor or source_height % self.factor:
            raise ValueError("Image size is not divisible by downsample factor")
        self.width = source_width // self.factor
        self.height = source_height // self.factor
        self.header_height = 104
        self.window_name = "Live SurgicalSAM2: distal and black shaft"
        self.stages = [
            (anchor_index, side, task)
            for anchor_index in range(len(manifest["anchors"]))
            for side in ("left", "right")
            for task in ("tips", "distal", "shaft")
        ]
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.args.output_dir / "annotation_state.json"
        self.state = self.load_or_initialize_state()
        self.stage_index = self.first_unaccepted_stage_index(
            int(self.state.get("next_stage_index", 0))
        )
        if self.stage_index >= len(self.stages):
            raise RuntimeError(
                "All anchors in this output directory are already complete"
            )
        self.image: np.ndarray | None = None
        self.points_positive: list[list[float]] = []
        self.points_negative: list[list[float]] = []
        self.candidate_masks: np.ndarray | None = None
        self.candidate_scores: np.ndarray | None = None
        self.selected_candidate = 0
        self.mask: np.ndarray | None = None
        self.other_mask: np.ndarray | None = None
        self.dirty = False
        self.load_stage()

    def load_or_initialize_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if payload.get("schema") != (
                "super_paper_lnd_live_sam2_anchor_masks_v4"
            ):
                raise RuntimeError(f"Unexpected schema in {self.state_path}")
            return payload
        state = {
            "schema": "super_paper_lnd_live_sam2_anchor_masks_v4",
            "completed": False,
            "tracking_image_size_wh": [self.width, self.height],
            "source_image_size_wh": self.manifest["image_size_wh"],
            "downsample_factor": self.factor,
            "checkpoint": str(self.args.checkpoint),
            "config": self.args.config,
            "next_stage_index": 0,
            "accepted": {},
            "draft": None,
        }
        if self.args.seed_annotation_dir is None:
            return state
        seed_state_path = (
            self.args.seed_annotation_dir / "annotation_state.json"
        )
        if not seed_state_path.is_file():
            raise FileNotFoundError(seed_state_path)
        seed = json.loads(seed_state_path.read_text(encoding="utf-8"))
        if seed.get("schema") != state["schema"]:
            raise RuntimeError(
                f"Unexpected seed schema in {seed_state_path}"
            )
        reused_slots: set[int] = set()
        for anchor_index, row in enumerate(self.manifest["anchors"]):
            slot = int(row["strict_pair_slot"])
            slot_complete = True
            for side in ("left", "right"):
                for task in ("tips", "distal", "shaft"):
                    key = f"slot_{slot:04d}_{side}_{task}"
                    record = seed.get("accepted", {}).get(key)
                    if record is None:
                        slot_complete = False
                        continue
                    copied = copy.deepcopy(record)
                    copied["anchor_index"] = anchor_index
                    state["accepted"][key] = copied
                    if task != "tips":
                        source = (
                            self.args.seed_annotation_dir
                            / str(copied["mask_path"])
                        )
                        destination = self.args.output_dir / source.name
                        if not source.is_file():
                            raise FileNotFoundError(source)
                        shutil.copy2(source, destination)
            if slot_complete:
                reused_slots.add(slot)
        state["seed_annotation_dir"] = str(
            self.args.seed_annotation_dir
        )
        state["reused_completed_slots"] = sorted(reused_slots)
        state["next_stage_index"] = self.first_unaccepted_stage_index(
            0,
            accepted=state["accepted"],
        )
        write_json(self.state_path, state)
        print(
            "Reused completed anchors from seed: "
            + ", ".join(map(str, sorted(reused_slots))),
            flush=True,
        )
        return state

    def first_unaccepted_stage_index(
        self,
        start: int,
        *,
        accepted: dict[str, Any] | None = None,
    ) -> int:
        records = self.state["accepted"] if accepted is None else accepted
        for index in range(max(0, start), len(self.stages)):
            if self.stage_key(*self.stages[index]) not in records:
                return index
        return len(self.stages)

    def stage_key(
        self,
        anchor_index: int,
        side: str,
        task: str,
    ) -> str:
        slot = int(
            self.manifest["anchors"][anchor_index]["strict_pair_slot"]
        )
        return f"slot_{slot:04d}_{side}_{task}"

    def mask_path(
        self,
        anchor_index: int,
        side: str,
        part: str,
    ) -> Path:
        return (
            self.args.output_dir
            / f"{self.stage_key(anchor_index, side, part)}_mask.png"
        )

    def save_state(self) -> None:
        write_json(self.state_path, self.state)

    def current_identity(self) -> tuple[int, str, str]:
        return self.stages[self.stage_index]

    def load_stage(self) -> None:
        anchor_index, side, task = self.current_identity()
        row = self.manifest["anchors"][anchor_index]
        image_path = self.root / row[f"{side}_image"]
        source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(image_path)
        self.image = cv2.resize(
            source,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )
        if task != "tips":
            self.predictor.set_image(
                cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            )
        self.points_positive = []
        self.points_negative = []
        self.candidate_masks = None
        self.candidate_scores = None
        self.mask = None
        self.selected_candidate = 0
        self.other_mask = None
        if task != "tips":
            other_part = "distal" if task == "shaft" else "shaft"
            other_path = self.mask_path(anchor_index, side, other_part)
            if other_path.exists():
                loaded = cv2.imread(str(other_path), cv2.IMREAD_GRAYSCALE)
                if loaded is not None:
                    self.other_mask = loaded > 0

        key = self.stage_key(anchor_index, side, task)
        record = self.state["accepted"].get(key)
        draft = self.state.get("draft")
        if (
            draft is not None
            and draft.get("stage_key") == key
        ):
            record = draft
        if record is not None:
            point_field = (
                "tip_points_xy"
                if task == "tips"
                else "positive_points_xy"
            )
            self.points_positive = [
                list(map(float, point))
                for point in record.get(point_field, [])
            ]
            self.points_negative = [
                list(map(float, point))
                for point in record.get("negative_points_xy", [])
            ]
            if self.points_positive and task != "tips":
                self.dirty = True
                self.update_prediction()

    def candidate_rank(self, index: int) -> tuple[float, ...]:
        assert self.candidate_masks is not None
        assert self.candidate_scores is not None
        candidate = self.candidate_masks[index]

        def prompt_hits(points: list[list[float]]) -> np.ndarray:
            if not points:
                return np.empty(0, dtype=bool)
            xy = np.rint(np.asarray(points)).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, self.width - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, self.height - 1)
            return candidate[xy[:, 1], xy[:, 0]]

        positive_misses = int(
            np.count_nonzero(~prompt_hits(self.points_positive))
        )
        negative_hits = int(
            np.count_nonzero(prompt_hits(self.points_negative))
        )
        overlap = (
            0
            if self.other_mask is None
            else int(np.count_nonzero(candidate & self.other_mask))
        )
        # Among candidates respecting clicks, prefer less overlap with the
        # already accepted physical part and then the smallest connected
        # interpretation.  This counters black-background takeover.
        return (
            float(positive_misses + negative_hits),
            float(overlap),
            float(candidate.sum()),
            -float(self.candidate_scores[index]),
        )

    def update_prediction(self) -> None:
        _anchor_index, _side, task = self.current_identity()
        if task == "tips":
            self.mask = None
            self.candidate_masks = None
            self.candidate_scores = None
            self.dirty = False
            self.save_draft()
            return
        if not self.points_positive:
            self.candidate_masks = None
            self.candidate_scores = None
            self.mask = None
            self.dirty = False
            return
        points = np.asarray(
            self.points_positive + self.points_negative,
            dtype=np.float32,
        )
        labels = np.asarray(
            [1] * len(self.points_positive)
            + [0] * len(self.points_negative),
            dtype=np.int64,
        )
        with torch.inference_mode(), torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            masks, scores, _logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )
        self.candidate_masks = np.asarray(masks).astype(bool)
        self.candidate_scores = np.asarray(scores, dtype=np.float64)
        self.selected_candidate = min(
            range(len(self.candidate_masks)),
            key=self.candidate_rank,
        )
        self.mask = self.candidate_masks[self.selected_candidate]
        self.dirty = False
        self.save_draft()

    def save_draft(self) -> None:
        anchor_index, side, task = self.current_identity()
        point_field = (
            "tip_points_xy"
            if task == "tips"
            else "positive_points_xy"
        )
        self.state["draft"] = {
            "stage_key": self.stage_key(anchor_index, side, task),
            point_field: self.points_positive,
            "negative_points_xy": self.points_negative,
        }
        self.save_state()

    def cycle_candidate(self) -> None:
        if self.candidate_masks is None:
            return
        self.selected_candidate = (
            self.selected_candidate + 1
        ) % len(self.candidate_masks)
        self.mask = self.candidate_masks[self.selected_candidate]
        self.save_draft()

    def reset_current(self) -> None:
        self.points_positive = []
        self.points_negative = []
        self.candidate_masks = None
        self.candidate_scores = None
        self.mask = None
        self.state["draft"] = None
        self.save_state()

    def undo(self) -> None:
        if self.points_negative:
            self.points_negative.pop()
        elif self.points_positive:
            self.points_positive.pop()
        self.dirty = True
        self.update_prediction()

    def accept(self) -> None:
        anchor_index, side, task = self.current_identity()
        if task == "tips":
            if len(self.points_positive) != 2:
                print("Exactly two jaw-tip points are required", flush=True)
                return
            key = self.stage_key(anchor_index, side, task)
            self.state["accepted"][key] = {
                "anchor_index": anchor_index,
                "strict_pair_slot": int(
                    self.manifest["anchors"][anchor_index][
                        "strict_pair_slot"
                    ]
                ),
                "side": side,
                "task": task,
                "tip_points_xy": self.points_positive,
                "tip_points_xy_full_resolution": (
                    np.asarray(self.points_positive) * self.factor
                ).tolist(),
            }
            self.finish_stage()
            return
        if self.mask is None or not self.points_positive:
            print("At least one positive point and a visible mask are required")
            return
        key = self.stage_key(anchor_index, side, task)
        output = self.mask_path(anchor_index, side, task)
        if not cv2.imwrite(
            str(output),
            self.mask.astype(np.uint8) * 255,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        ):
            raise RuntimeError(f"Failed to write {output}")
        score = (
            None
            if self.candidate_scores is None
            else float(self.candidate_scores[self.selected_candidate])
        )
        self.state["accepted"][key] = {
            "anchor_index": anchor_index,
            "strict_pair_slot": int(
                self.manifest["anchors"][anchor_index]["strict_pair_slot"]
            ),
            "side": side,
            "task": task,
            "positive_points_xy": self.points_positive,
            "negative_points_xy": self.points_negative,
            "selected_candidate": self.selected_candidate,
            "predicted_iou_score": score,
            "mask_area_px": int(self.mask.sum()),
            "mask_path": output.name,
        }
        self.finish_stage()

    def finish_stage(self) -> None:
        self.state["draft"] = None
        next_stage_index = self.first_unaccepted_stage_index(
            self.stage_index + 1
        )
        if next_stage_index == len(self.stages):
            self.state["completed"] = True
            self.state["next_stage_index"] = len(self.stages)
            self.save_state()
            self.write_overview()
            cv2.destroyAllWindows()
            print(f"Completed live SAM2 masks: {self.state_path}")
            raise StopIteration
        self.stage_index = next_stage_index
        self.state["next_stage_index"] = self.stage_index
        self.save_state()
        self.load_stage()

    def go_back(self) -> None:
        if self.stage_index == 0:
            return
        self.stage_index -= 1
        self.state["next_stage_index"] = self.stage_index
        self.state["completed"] = False
        self.state["draft"] = None
        self.save_state()
        self.load_stage()

    def write_overview(self) -> None:
        rows = []
        for anchor_index, row in enumerate(self.manifest["anchors"]):
            panels = []
            for side in ("left", "right"):
                source = cv2.imread(
                    str(self.root / row[f"{side}_image"]),
                    cv2.IMREAD_COLOR,
                )
                if source is None:
                    raise FileNotFoundError(row[f"{side}_image"])
                shaft = (
                    cv2.imread(
                        str(self.mask_path(anchor_index, side, "shaft")),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    > 0
                )
                distal = (
                    cv2.imread(
                        str(self.mask_path(anchor_index, side, "distal")),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    > 0
                )
                shaft = cv2.resize(
                    shaft.astype(np.uint8),
                    (source.shape[1], source.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                distal = cv2.resize(
                    distal.astype(np.uint8),
                    (source.shape[1], source.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                overlay = tint_mask(source, shaft, PART_COLORS["shaft"])
                overlay = tint_mask(overlay, distal, PART_COLORS["distal"])
                tips_key = self.stage_key(anchor_index, side, "tips")
                tip_record = self.state["accepted"].get(tips_key)
                if tip_record is None:
                    raise RuntimeError(f"Missing accepted tips: {tips_key}")
                for index, point in enumerate(
                    tip_record["tip_points_xy_full_resolution"]
                ):
                    xy = tuple(np.rint(point).astype(int))
                    color = (
                        (40, 255, 40) if index == 0 else (255, 40, 255)
                    )
                    cv2.circle(
                        overlay,
                        xy,
                        9,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        overlay,
                        xy,
                        12,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (overlay.shape[1], 38),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    overlay,
                    (
                        f"slot {int(row['strict_pair_slot']):04d} "
                        f"{side}: blue shaft, orange distal"
                    ),
                    (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                scale = 480.0 / overlay.shape[1]
                panels.append(
                    cv2.resize(
                        overlay,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_AREA,
                    )
                )
            rows.append(np.concatenate(panels, axis=1))
        output = self.args.output_dir / "live_sam2_anchor_overview.png"
        if not cv2.imwrite(str(output), np.concatenate(rows, axis=0)):
            raise RuntimeError(f"Failed to write {output}")
        self.state["overview"] = str(output)
        self.save_state()

    def mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        _flags: int,
        _parameter: object,
    ) -> None:
        image_y = y - self.header_height
        if not (0 <= x < self.width and 0 <= image_y < self.height):
            return
        _anchor_index, _side, task = self.current_identity()
        if task == "tips":
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.points_positive) < 2:
                    self.points_positive.append([float(x), float(image_y)])
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.points_positive:
                    self.points_positive.pop()
            else:
                return
            self.dirty = True
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points_positive.append([float(x), float(image_y)])
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points_negative.append([float(x), float(image_y)])
        else:
            return
        self.dirty = True

    def render(self) -> np.ndarray:
        assert self.image is not None
        view = self.image.copy()
        anchor_index, side, task = self.current_identity()
        if self.other_mask is not None and task != "tips":
            other_part = "distal" if task == "shaft" else "shaft"
            view = tint_mask(
                view,
                self.other_mask,
                PART_COLORS[other_part],
                alpha=0.25,
            )
        if self.mask is not None and task != "tips":
            view = tint_mask(view, self.mask, PART_COLORS[task])
        for index, point in enumerate(self.points_positive):
            xy = tuple(np.rint(point).astype(int))
            if task == "tips":
                color = (40, 255, 40) if index == 0 else (255, 40, 255)
            else:
                color = (40, 240, 40)
            cv2.circle(view, xy, 6, color, -1, cv2.LINE_AA)
            cv2.circle(view, xy, 8, (0, 0, 0), 2, cv2.LINE_AA)
        for point in self.points_negative if task != "tips" else []:
            xy = tuple(np.rint(point).astype(int))
            cv2.drawMarker(
                view,
                xy,
                (30, 30, 255),
                cv2.MARKER_TILTED_CROSS,
                16,
                3,
                cv2.LINE_AA,
            )
        canvas = np.zeros(
            (self.height + self.header_height, self.width, 3),
            dtype=np.uint8,
        )
        canvas[self.header_height :] = view
        row = self.manifest["anchors"][anchor_index]
        score = (
            "none"
            if self.candidate_scores is None
            else (
                f"{float(self.candidate_scores[self.selected_candidate]):.3f}"
            )
        )
        area = 0 if self.mask is None else int(self.mask.sum())
        title = (
            f"{self.stage_index + 1}/{len(self.stages)} | "
            f"slot {int(row['strict_pair_slot']):04d} {side} | "
            + (
                f"mark 2 jaw tips ({len(self.points_positive)}/2)"
                if task == "tips"
                else (
                    f"segment {task} | "
                    f"candidate {self.selected_candidate + 1}/3 "
                    f"score {score} area {area}"
                )
            )
        )
        help_text = (
            (
                "LEFT=jaw tip (exactly 2)  U=undo  R=reset  "
                "ENTER=accept  B=previous  Q=save+quit"
            )
            if task == "tips"
            else (
                "LEFT=foreground  RIGHT=background  M=next candidate  "
                "U=undo  R=reset  ENTER=accept  B=previous  Q=save+quit"
            )
        )
        cv2.putText(
            canvas,
            title,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            PART_COLORS[task],
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            help_text,
            (12, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                "Click the two physical jaw tips, then press ENTER."
                if task == "tips"
                else "Confirm the colored REGION, not the number of clicks."
            ),
            (12, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.window_name,
            self.args.window_width,
            self.args.window_height,
        )
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        try:
            while True:
                if self.dirty:
                    self.update_prediction()
                cv2.imshow(self.window_name, self.render())
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("m"), ord("M")):
                    self.cycle_candidate()
                elif key in (ord("u"), ord("U")):
                    self.undo()
                elif key in (ord("r"), ord("R")):
                    self.reset_current()
                elif key in (13, 10, ord("s"), ord("S")):
                    self.accept()
                elif key in (ord("b"), ord("B")):
                    self.go_back()
                elif key in (ord("q"), ord("Q"), 27):
                    self.save_draft()
                    cv2.destroyAllWindows()
                    print(f"Saved live SAM2 progress: {self.state_path}")
                    return
        except StopIteration:
            return


def main() -> None:
    args = parse_args()
    manifest_path = args.root / "prompt_manifest.json"
    for path in (manifest_path, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != (
        "super_paper_lnd_stereo_multianchor_prompt_manifest_v2"
    ):
        raise RuntimeError(f"Unexpected schema in {manifest_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.cuda_device)
    print(f"Loading SurgicalSAM2 on cuda:{args.cuda_device}", flush=True)
    model = build_sam2(args.config, str(args.checkpoint))
    predictor = Torch26SAM2ImagePredictor(model)
    annotator = LiveMaskAnnotator(
        args=args,
        manifest=manifest,
        predictor=predictor,
    )
    annotator.run()


if __name__ == "__main__":
    main()
