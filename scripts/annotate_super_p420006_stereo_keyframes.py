#!/usr/bin/env python3
"""Annotate rectified stereo instrument keyframes.

The annotator shows the left and right raw rectified images together.  It
stores full-resolution polygons for the tool body and the two jaws, plus the
two jaw-tip pixels.  No historical pose, mask, or image-derived asset is read.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_p420006_stereo_v1"
)
P420_ANNOTATION_SCHEMA = "super_p420006_stereo_manual_annotation_v1"
PAPER_ANNOTATION_SCHEMA = "super_paper_lnd_stereo_manual_annotation_v1"
ACCEPTED_MANIFEST_SCHEMAS = {
    "super_p420006_stereo_keyframe_manifest_v1",
    "super_paper_lnd_stereo_first_frame_manifest_v1",
}
CLASS_NAMES = {1: "body", 2: "jaw_1", 3: "jaw_2"}
CLASS_COLORS = {
    "body": (60, 220, 60),
    "jaw_1": (40, 90, 255),
    "jaw_2": (255, 170, 30),
}
TIP_NAMES = {4: "jaw_1", 5: "jaw_2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label P420006 body/jaws and jaw tips in stereo anchors."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--keyframe", type=int, default=0)
    parser.add_argument("--window-width", type=int, default=1800)
    parser.add_argument("--window-height", type=int, default=950)
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def empty_view() -> dict[str, Any]:
    return {
        "polygons": {name: [] for name in CLASS_NAMES.values()},
        "tips": {
            name: {"visible": None, "point_xy": None}
            for name in TIP_NAMES.values()
        },
    }


def empty_annotation(
    row: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "schema": row.get("annotation_schema", P420_ANNOTATION_SCHEMA),
        "keyframe_index": int(row["keyframe_index"]),
        "strict_pair_slot": int(row["strict_pair_slot"]),
        "image_size_wh": [width, height],
        "coordinate_frame": "rectified image pixels, origin top-left",
        "class_map": {
            "0": "background",
            "1": "body",
            "2": "jaw_1",
            "3": "jaw_2",
        },
        "instructions": (
            "body excludes jaw pixels; in each keyframe jaw_1 is the "
            "image-left jaw in the LEFT view, and the same physical jaw must "
            "be labelled jaw_1 in the RIGHT view"
        ),
        "source_images": {
            "left": {
                "path": row["left_image"],
                "sha256": row["left_image_sha256"],
            },
            "right": {
                "path": row["right_image"],
                "sha256": row["right_image_sha256"],
            },
        },
        "views": {"left": empty_view(), "right": empty_view()},
    }


def annotation_path(root: Path, keyframe_index: int) -> Path:
    return root / "annotations" / f"keyframe_{keyframe_index:02d}.json"


def load_annotation(
    root: Path,
    row: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    path = annotation_path(root, int(row["keyframe_index"]))
    if not path.exists():
        return empty_annotation(row, width, height)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_schema = row.get(
        "annotation_schema",
        P420_ANNOTATION_SCHEMA,
    )
    if payload.get("schema") != expected_schema:
        raise RuntimeError(f"Unexpected annotation schema in {path}")
    if payload.get("keyframe_index") != row["keyframe_index"]:
        raise RuntimeError(f"Keyframe identity mismatch in {path}")
    if payload.get("image_size_wh") != [width, height]:
        raise RuntimeError(f"Image size mismatch in {path}")
    for side in ("left", "right"):
        source = root / payload["source_images"][side]["path"]
        if sha256(source) != payload["source_images"][side]["sha256"]:
            raise RuntimeError(f"Source image hash mismatch for {source}")
    return payload


def rasterize_view(
    view: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    labels = np.zeros((height, width), dtype=np.uint8)
    for class_id, class_name in CLASS_NAMES.items():
        for polygon in view["polygons"][class_name]:
            points = np.rint(np.asarray(polygon, dtype=np.float64)).astype(
                np.int32
            )
            if len(points) >= 3:
                cv2.fillPoly(labels, [points], class_id)
    return labels


def save_annotation(root: Path, annotation: dict[str, Any]) -> None:
    keyframe_index = int(annotation["keyframe_index"])
    width, height = annotation["image_size_wh"]
    path = annotation_path(root, keyframe_index)
    path.write_text(
        json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for side in ("left", "right"):
        labels = rasterize_view(
            annotation["views"][side],
            int(width),
            int(height),
        )
        output_path = (
            root
            / "annotations"
            / f"keyframe_{keyframe_index:02d}_{side}_labels.png"
        )
        if not cv2.imwrite(
            str(output_path),
            labels,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        ):
            raise RuntimeError(f"Failed to write {output_path}")


def view_complete(view: dict[str, Any]) -> bool:
    polygons_complete = all(
        len(view["polygons"][name]) > 0 for name in CLASS_NAMES.values()
    )
    tips_complete = all(
        view["tips"][name]["visible"] is not None
        and (
            not view["tips"][name]["visible"]
            or view["tips"][name]["point_xy"] is not None
        )
        for name in TIP_NAMES.values()
    )
    return polygons_complete and tips_complete


class StereoAnnotator:
    def __init__(
        self,
        root: Path,
        rows: list[dict[str, Any]],
        start: int,
        window_width: int,
        window_height: int,
    ) -> None:
        self.root = root
        self.rows = rows
        self.index = int(np.clip(start, 0, len(rows) - 1))
        self.window_width = window_width
        self.window_height = window_height
        annotation_schema = rows[0].get(
            "annotation_schema",
            P420_ANNOTATION_SCHEMA,
        )
        instrument = (
            "paper LND"
            if annotation_schema == PAPER_ANNOTATION_SCHEMA
            else "P420006"
        )
        self.window_name = f"{instrument} stereo manual annotation"
        self.mode = 1
        self.current_points: list[list[float]] = []
        self.current_side: str | None = None
        self.hover_side = "left"
        self.hover_xy: tuple[float, float] | None = None
        self.histories: dict[int, list[dict[str, Any]]] = {}
        self.images: dict[str, np.ndarray] = {}
        self.annotation: dict[str, Any] = {}
        self.scale = 1.0
        self.view_width = 0
        self.view_height = 0
        self.image_top = 100
        self.load_current()

    def load_current(self) -> None:
        row = self.rows[self.index]
        self.images = {
            side: cv2.imread(
                str(self.root / row[f"{side}_image"]),
                cv2.IMREAD_COLOR,
            )
            for side in ("left", "right")
        }
        if any(image is None for image in self.images.values()):
            raise RuntimeError(f"Failed to load keyframe {self.index}")
        height, width = self.images["left"].shape[:2]
        if self.images["right"].shape[:2] != (height, width):
            raise RuntimeError("Stereo image sizes disagree")
        self.scale = min(
            self.window_width / (2.0 * width),
            (self.window_height - self.image_top) / height,
            1.0,
        )
        self.view_width = int(round(width * self.scale))
        self.view_height = int(round(height * self.scale))
        self.annotation = load_annotation(
            self.root,
            row,
            width,
            height,
        )
        self.current_points = []
        self.current_side = None
        self.hover_xy = None

    def push_history(self) -> None:
        self.histories.setdefault(self.index, []).append(
            copy.deepcopy(self.annotation)
        )
        self.histories[self.index] = self.histories[self.index][-50:]

    def map_mouse(
        self,
        x: int,
        y: int,
    ) -> tuple[str, tuple[float, float]] | None:
        if y < self.image_top or y >= self.image_top + self.view_height:
            return None
        if x < 0 or x >= 2 * self.view_width:
            return None
        side = "left" if x < self.view_width else "right"
        x_local = x if side == "left" else x - self.view_width
        full_x = float(x_local / self.scale)
        full_y = float((y - self.image_top) / self.scale)
        width, height = self.annotation["image_size_wh"]
        full_x = float(np.clip(full_x, 0.0, width - 1.0))
        full_y = float(np.clip(full_y, 0.0, height - 1.0))
        return side, (full_x, full_y)

    def mouse(self, event: int, x: int, y: int, _flags: int, _data: Any) -> None:
        mapped = self.map_mouse(x, y)
        if mapped is not None:
            self.hover_side, self.hover_xy = mapped
        if event != cv2.EVENT_LBUTTONDOWN or mapped is None:
            return
        side, point = mapped
        if self.mode in CLASS_NAMES:
            if self.current_side is None:
                self.current_side = side
            if side == self.current_side:
                self.current_points.append([point[0], point[1]])
            return
        tip_name = TIP_NAMES[self.mode]
        self.push_history()
        self.annotation["views"][side]["tips"][tip_name] = {
            "visible": True,
            "point_xy": [point[0], point[1]],
        }

    def commit_polygon(self) -> None:
        if self.mode not in CLASS_NAMES:
            return
        if self.current_side is None or len(self.current_points) < 3:
            return
        self.push_history()
        class_name = CLASS_NAMES[self.mode]
        self.annotation["views"][self.current_side]["polygons"][
            class_name
        ].append(copy.deepcopy(self.current_points))
        self.current_points = []
        self.current_side = None

    def mark_tip_invisible(self) -> None:
        if self.mode not in TIP_NAMES:
            return
        tip_name = TIP_NAMES[self.mode]
        self.push_history()
        self.annotation["views"][self.hover_side]["tips"][tip_name] = {
            "visible": False,
            "point_xy": None,
        }

    def delete_active(self) -> None:
        self.push_history()
        if self.mode in CLASS_NAMES:
            polygons = self.annotation["views"][self.hover_side]["polygons"][
                CLASS_NAMES[self.mode]
            ]
            if polygons:
                polygons.pop()
        else:
            self.annotation["views"][self.hover_side]["tips"][
                TIP_NAMES[self.mode]
            ] = {"visible": None, "point_xy": None}

    def undo(self) -> None:
        if self.current_points:
            self.current_points.pop()
            if not self.current_points:
                self.current_side = None
            return
        history = self.histories.get(self.index, [])
        if history:
            self.annotation = history.pop()

    def navigate(self, increment: int) -> None:
        if self.current_points:
            return
        save_annotation(self.root, self.annotation)
        self.index = int(np.clip(self.index + increment, 0, len(self.rows) - 1))
        self.load_current()

    def draw_view(
        self,
        side: str,
        x_offset: int,
        canvas: np.ndarray,
    ) -> None:
        image = cv2.resize(
            self.images[side],
            (self.view_width, self.view_height),
            interpolation=cv2.INTER_AREA,
        )
        overlay = image.copy()
        view = self.annotation["views"][side]
        for class_name, color in CLASS_COLORS.items():
            for polygon in view["polygons"][class_name]:
                points = np.rint(
                    np.asarray(polygon, dtype=np.float64) * self.scale
                ).astype(np.int32)
                if len(points) >= 3:
                    cv2.fillPoly(overlay, [points], color)
                    cv2.polylines(
                        image,
                        [points],
                        True,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
        image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0.0)
        for tip_name, tip in view["tips"].items():
            if tip["visible"] and tip["point_xy"] is not None:
                point = tuple(
                    np.rint(
                        np.asarray(tip["point_xy"], dtype=np.float64)
                        * self.scale
                    ).astype(int)
                )
                color = CLASS_COLORS[tip_name]
                cv2.drawMarker(
                    image,
                    point,
                    color,
                    cv2.MARKER_CROSS,
                    18,
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    "T1" if tip_name == "jaw_1" else "T2",
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if self.current_side == side and self.current_points:
            points = np.rint(
                np.asarray(self.current_points) * self.scale
            ).astype(np.int32)
            color = CLASS_COLORS[CLASS_NAMES[self.mode]]
            if len(points) >= 2:
                cv2.polylines(image, [points], False, color, 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(image, tuple(point), 4, color, -1, cv2.LINE_AA)

        # Rectified stereo cue: the current mouse row is drawn in both views.
        if self.hover_xy is not None:
            y_line = int(round(self.hover_xy[1] * self.scale))
            cv2.line(
                image,
                (0, y_line),
                (self.view_width - 1, y_line),
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            image,
            f"{side.upper()}  {'OK' if view_complete(view) else 'INCOMPLETE'}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (40, 255, 40) if view_complete(view) else (0, 190, 255),
            2,
            cv2.LINE_AA,
        )
        canvas[
            self.image_top : self.image_top + self.view_height,
            x_offset : x_offset + self.view_width,
        ] = image

    def render(self) -> np.ndarray:
        canvas = np.zeros(
            (self.image_top + self.view_height, 2 * self.view_width, 3),
            dtype=np.uint8,
        )
        row = self.rows[self.index]
        mode_name = (
            CLASS_NAMES[self.mode]
            if self.mode in CLASS_NAMES
            else f"{TIP_NAMES[self.mode]} tip"
        )
        header = (
            f"KF {self.index:02d}/{len(self.rows)-1:02d}  "
            f"t={row['pair_time_s']:.3f}s  dt={row['stereo_delta_ms']:+.2f}ms  "
            f"MODE={mode_name}"
        )
        help_text = (
            "1 body  2 jaw1  3 jaw2  4 tip1  5 tip2 | click: add/set | "
            "Enter: close polygon | U/Backspace: undo | D: delete | "
            "X: tip invisible | P/N: prev/next | S: save | Q: save+quit"
        )
        cv2.putText(
            canvas,
            header,
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            help_text,
            (12, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )
        self.draw_view("left", 0, canvas)
        self.draw_view("right", self.view_width, canvas)
        cv2.line(
            canvas,
            (self.view_width, self.image_top),
            (self.view_width, self.image_top + self.view_height - 1),
            (255, 255, 255),
            1,
        )
        return canvas

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.window_name,
            2 * self.view_width,
            self.image_top + self.view_height,
        )
        cv2.setMouseCallback(self.window_name, self.mouse)
        while True:
            cv2.imshow(self.window_name, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in (ord("1"), ord("2"), ord("3")):
                if not self.current_points:
                    self.mode = int(chr(key))
            elif key in (ord("4"), ord("5")):
                if not self.current_points:
                    self.mode = int(chr(key))
            elif key in (10, 13):
                self.commit_polygon()
            elif key in (8, 127, ord("u"), ord("U")):
                self.undo()
            elif key in (ord("d"), ord("D")):
                if not self.current_points:
                    self.delete_active()
            elif key in (ord("x"), ord("X")):
                if not self.current_points:
                    self.mark_tip_invisible()
            elif key in (ord("s"), ord("S")):
                if not self.current_points:
                    save_annotation(self.root, self.annotation)
            elif key in (ord("n"), ord("N")):
                self.navigate(1)
            elif key in (ord("p"), ord("P")):
                self.navigate(-1)
            elif key in (ord("q"), ord("Q"), 27):
                if not self.current_points:
                    save_annotation(self.root, self.annotation)
                    break
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    manifest_path = args.root / "pair_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in ACCEPTED_MANIFEST_SCHEMAS:
        raise RuntimeError("Unexpected keyframe manifest schema")
    rows = manifest["keyframes"]
    if not rows:
        raise RuntimeError("Manifest contains no annotation frames")
    args.root.joinpath("annotations").mkdir(exist_ok=True)
    StereoAnnotator(
        args.root,
        rows,
        args.keyframe,
        args.window_width,
        args.window_height,
    ).run()


if __name__ == "__main__":
    main()
