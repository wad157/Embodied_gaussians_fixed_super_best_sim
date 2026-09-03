#!/usr/bin/env python3
"""Interactively place stereo shaft/distal SAM2 prompts at several anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_multianchor_v2"
)
PROMPT_KEYS = (
    "shaft_positive",
    "shaft_negative",
    "distal_positive",
    "distal_negative",
)
COLORS = {
    "shaft_positive": (40, 240, 40),
    "shaft_negative": (30, 30, 255),
    "distal_positive": (255, 220, 20),
    "distal_negative": (255, 40, 220),
}
MODE_KEYS = {
    ord("1"): "shaft_positive",
    ord("2"): "shaft_negative",
    ord("3"): "distal_positive",
    ord("4"): "distal_negative",
}
MINIMUM_COUNTS = {
    "shaft_positive": 4,
    "shaft_negative": 2,
    "distal_positive": 2,
    "distal_negative": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Place positive/negative point prompts for separate black-shaft "
            "and silver-distal SAM2 objects in paired rectified images."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--window-width", type=int, default=1800)
    parser.add_argument("--window-height", type=int, default=950)
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def empty_points() -> dict[str, list[list[float]]]:
    return {name: [] for name in PROMPT_KEYS}


class PromptAnnotator:
    def __init__(
        self,
        *,
        root: Path,
        manifest: dict[str, Any],
        window_width: int,
        window_height: int,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.width, self.height = manifest["image_size_wh"]
        self.window_width = window_width
        self.window_height = window_height
        self.window_name = "paper LND multi-anchor SAM2 prompts"
        self.anchor_index = 0
        self.mode = "shaft_positive"
        self.history: list[tuple[int, str, str]] = []
        self.annotations = self.load_or_initialize()
        self.images = self.load_images()
        self.display_scale = min(
            window_width / (2.0 * self.width),
            (window_height - 95.0) / self.height,
        )
        self.panel_width = int(round(self.width * self.display_scale))
        self.panel_height = int(round(self.height * self.display_scale))

    @property
    def output_path(self) -> Path:
        return self.root / "annotations/multianchor_point_prompts.json"

    def load_or_initialize(self) -> dict[str, Any]:
        if self.output_path.exists():
            payload = json.loads(self.output_path.read_text(encoding="utf-8"))
            if payload.get("schema") != (
                "super_paper_lnd_stereo_multianchor_point_prompts_v2"
            ):
                raise RuntimeError(f"Unexpected schema in {self.output_path}")
            return payload
        anchors = []
        for row in self.manifest["anchors"]:
            anchors.append(
                {
                    "anchor_index": int(row["anchor_index"]),
                    "strict_pair_slot": int(row["strict_pair_slot"]),
                    "views": {
                        "left": empty_points(),
                        "right": empty_points(),
                    },
                }
            )
        return {
            "schema": "super_paper_lnd_stereo_multianchor_point_prompts_v2",
            "coordinate_frame": self.manifest["coordinate_frame"],
            "image_size_wh": [self.width, self.height],
            "instructions": {
                "shaft_positive": (
                    "points on the dark shaft, spread along its visible length"
                ),
                "shaft_negative": (
                    "background points immediately on both sides of the shaft"
                ),
                "distal_positive": (
                    "points on the silver wrist housing and jaws"
                ),
                "distal_negative": (
                    "background points around the distal tool"
                ),
            },
            "anchors": anchors,
            "completed": False,
        }

    def load_images(self) -> list[dict[str, np.ndarray]]:
        output = []
        for row in self.manifest["anchors"]:
            pair = {}
            for side in ("left", "right"):
                path = self.root / row[f"{side}_image"]
                if sha256(path) != row[f"{side}_image_sha256"]:
                    raise RuntimeError(f"Image hash mismatch: {path}")
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(path)
                pair[side] = image
            output.append(pair)
        return output

    def current_anchor(self) -> dict[str, Any]:
        return self.annotations["anchors"][self.anchor_index]

    def save(self, completed: bool = False) -> None:
        self.annotations["completed"] = completed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(self.annotations, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def map_click(self, x: int, y: int) -> tuple[str, list[float]] | None:
        image_y = y - 82
        if image_y < 0 or image_y >= self.panel_height:
            return None
        if 0 <= x < self.panel_width:
            side = "left"
            image_x = x
        elif self.panel_width <= x < 2 * self.panel_width:
            side = "right"
            image_x = x - self.panel_width
        else:
            return None
        return (
            side,
            [
                float(np.clip(image_x / self.display_scale, 0, self.width - 1)),
                float(np.clip(image_y / self.display_scale, 0, self.height - 1)),
            ],
        )

    def on_mouse(
        self,
        event: int,
        x: int,
        y: int,
        _flags: int,
        _parameter: object,
    ) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        mapped = self.map_click(x, y)
        if mapped is None:
            return
        side, point = mapped
        self.current_anchor()["views"][side][self.mode].append(point)
        self.history.append((self.anchor_index, side, self.mode))

    def undo(self) -> None:
        if not self.history:
            return
        anchor_index, side, mode = self.history.pop()
        points = self.annotations["anchors"][anchor_index]["views"][side][mode]
        if points:
            points.pop()

    def clear_mode(self) -> None:
        for side in ("left", "right"):
            self.current_anchor()["views"][side][self.mode] = []

    def missing_requirements(self) -> list[str]:
        missing = []
        for anchor in self.annotations["anchors"]:
            for side in ("left", "right"):
                for name, minimum in MINIMUM_COUNTS.items():
                    count = len(anchor["views"][side][name])
                    if count < minimum:
                        missing.append(
                            f"slot {anchor['strict_pair_slot']} "
                            f"{side} {name}: {count}/{minimum}"
                        )
        return missing

    def render(self) -> np.ndarray:
        pair = self.images[self.anchor_index]
        panels = []
        anchor = self.current_anchor()
        for side in ("left", "right"):
            panel = cv2.resize(
                pair[side],
                (self.panel_width, self.panel_height),
                interpolation=cv2.INTER_AREA,
            )
            for name in PROMPT_KEYS:
                color = COLORS[name]
                positive = name.endswith("positive")
                for point in anchor["views"][side][name]:
                    xy = tuple(
                        np.rint(
                            np.asarray(point) * self.display_scale
                        ).astype(int)
                    )
                    if positive:
                        cv2.circle(panel, xy, 6, color, -1, cv2.LINE_AA)
                        cv2.circle(panel, xy, 8, (0, 0, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.drawMarker(
                            panel,
                            xy,
                            color,
                            cv2.MARKER_TILTED_CROSS,
                            14,
                            3,
                            cv2.LINE_AA,
                        )
            cv2.putText(
                panel,
                side.upper(),
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)
        canvas = np.zeros(
            (self.panel_height + 82, 2 * self.panel_width, 3),
            dtype=np.uint8,
        )
        canvas[82:] = np.concatenate(panels, axis=1)
        row = self.manifest["anchors"][self.anchor_index]
        title = (
            f"anchor {self.anchor_index + 1}/{len(self.images)} | "
            f"strict slot {row['strict_pair_slot']} | mode: {self.mode}"
        )
        help_line = (
            "1 shaft+  2 shaft-  3 distal+  4 distal-  | "
            "left click add  u undo  c clear mode  [/] prev/next  s finish"
        )
        cv2.putText(
            canvas,
            title,
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            COLORS[self.mode],
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            help_line,
            (12, 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.window_name,
            self.window_width,
            self.window_height,
        )
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        while True:
            cv2.imshow(self.window_name, self.render())
            key = cv2.waitKey(30) & 0xFF
            if key in MODE_KEYS:
                self.mode = MODE_KEYS[key]
            elif key in (ord("u"), ord("U")):
                self.undo()
            elif key in (ord("c"), ord("C")):
                self.clear_mode()
            elif key in (ord("]"), ord("n"), ord("N")):
                self.save(completed=False)
                self.anchor_index = min(
                    self.anchor_index + 1,
                    len(self.images) - 1,
                )
            elif key in (ord("["), ord("p"), ord("P")):
                self.save(completed=False)
                self.anchor_index = max(self.anchor_index - 1, 0)
            elif key in (ord("s"), ord("S")):
                missing = self.missing_requirements()
                if missing:
                    print("Cannot finish; missing prompts:", flush=True)
                    for line in missing:
                        print(f"  {line}", flush=True)
                    continue
                self.save(completed=True)
                cv2.destroyAllWindows()
                print(f"Saved completed prompts: {self.output_path}")
                return
            elif key in (ord("q"), ord("Q"), 27):
                self.save(completed=False)
                cv2.destroyAllWindows()
                print(f"Saved incomplete prompts: {self.output_path}")
                return


def main() -> None:
    args = parse_args()
    manifest_path = args.root / "prompt_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != (
        "super_paper_lnd_stereo_multianchor_prompt_manifest_v2"
    ):
        raise RuntimeError(f"Unexpected schema in {manifest_path}")
    annotator = PromptAnnotator(
        root=args.root,
        manifest=manifest,
        window_width=args.window_width,
        window_height=args.window_height,
    )
    annotator.run()


if __name__ == "__main__":
    main()
