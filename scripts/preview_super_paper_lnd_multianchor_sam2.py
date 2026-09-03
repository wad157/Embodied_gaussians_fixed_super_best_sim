#!/usr/bin/env python3
"""Preview separate shaft/distal SurgicalSAM2 masks at prompt anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from track_super_p420006_stereo_surgicalsam2 import build_predictor


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_paper_lnd_stereo_multianchor_v2"
)
PAPER_REPO = Path("/Media_HDD/jwshan/wad/online_dvrk_tracking")
CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_s.yaml"
CHECKPOINT_NAME = "sam2.1_hiera_s_endo18.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate manual multi-anchor point prompts with separate shaft "
            "and distal SurgicalSAM2 objects, without temporal propagation."
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
        default=DEFAULT_ROOT / "previews/multianchor_sam2_prompt_test",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def object_prompts(
    view: dict[str, list[list[float]]],
    *,
    object_name: str,
    downsample_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    other_name = "distal" if object_name == "shaft" else "shaft"
    positive = np.asarray(
        view[f"{object_name}_positive"],
        dtype=np.float32,
    ).reshape(-1, 2)
    explicit_negative = np.asarray(
        view[f"{object_name}_negative"],
        dtype=np.float32,
    ).reshape(-1, 2)
    # Positive points on the other physical part are reliable cross-object
    # negatives and prevent the two masks from collapsing into one object.
    cross_object_negative = np.asarray(
        view[f"{other_name}_positive"],
        dtype=np.float32,
    ).reshape(-1, 2)
    negative = np.concatenate(
        [explicit_negative, cross_object_negative],
        axis=0,
    )
    points = np.concatenate([positive, negative], axis=0)
    labels = np.concatenate(
        [
            np.ones(len(positive), dtype=np.int64),
            np.zeros(len(negative), dtype=np.int64),
        ]
    )
    return points / float(downsample_factor), labels


def unpack_logits(logits: torch.Tensor) -> np.ndarray:
    masks = logits.detach().to(torch.float32).cpu().numpy()
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise RuntimeError(f"Unexpected mask-logit shape {tuple(logits.shape)}")
    return masks > 0


def tint_parts(
    image: np.ndarray,
    shaft: np.ndarray,
    distal: np.ndarray,
) -> np.ndarray:
    output = image.copy()
    overlay = image.copy()
    shaft_only = shaft & ~distal
    distal_only = distal & ~shaft
    overlap = shaft & distal
    overlay[shaft_only] = (255, 120, 20)
    overlay[distal_only] = (30, 190, 255)
    overlay[overlap] = (220, 40, 220)
    active = shaft | distal
    output[active] = cv2.addWeighted(
        image[active],
        0.52,
        overlay[active],
        0.48,
        0.0,
    )
    for mask, color in (
        (shaft, (255, 120, 20)),
        (distal, (30, 190, 255)),
    ):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)
    return output


def draw_prompts(
    image: np.ndarray,
    view: dict[str, list[list[float]]],
) -> np.ndarray:
    output = image.copy()
    colors = {
        "shaft_positive": (40, 240, 40),
        "shaft_negative": (30, 30, 255),
        "distal_positive": (255, 220, 20),
        "distal_negative": (255, 40, 220),
    }
    for name, points in view.items():
        positive = name.endswith("positive")
        for point in points:
            xy = tuple(np.rint(point).astype(int))
            if positive:
                cv2.circle(output, xy, 5, colors[name], -1, cv2.LINE_AA)
                cv2.circle(output, xy, 7, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.drawMarker(
                    output,
                    xy,
                    colors[name],
                    cv2.MARKER_TILTED_CROSS,
                    12,
                    2,
                    cv2.LINE_AA,
                )
    return output


def prompt_metrics(
    mask: np.ndarray,
    *,
    positive: list[list[float]],
    negative: list[list[float]],
) -> dict[str, Any]:
    height, width = mask.shape

    def hits(points: list[list[float]]) -> np.ndarray:
        xy = np.rint(np.asarray(points, dtype=np.float64)).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
        return mask[xy[:, 1], xy[:, 0]]

    positive_hits = hits(positive)
    negative_hits = hits(negative)
    y_values, x_values = np.where(mask)
    bbox = (
        None
        if len(x_values) == 0
        else [
            int(x_values.min()),
            int(y_values.min()),
            int(x_values.max()),
            int(y_values.max()),
        ]
    )
    return {
        "area_px": int(mask.sum()),
        "bbox_xyxy": bbox,
        "positive_hit_fraction": float(positive_hits.mean()),
        "negative_rejection_fraction": float((~negative_hits).mean()),
    }


def add_title(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    args = parse_args()
    manifest_path = args.root / "prompt_manifest.json"
    prompts_path = (
        args.root / "annotations/multianchor_point_prompts.json"
    )
    for path in (manifest_path, prompts_path, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    if prompts.get("completed") is not True:
        raise RuntimeError("Multi-anchor prompts have not been completed")
    if len(manifest["anchors"]) != len(prompts["anchors"]):
        raise RuntimeError("Prompt/manifest anchor count mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.cuda_device)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} is not empty; pass --overwrite"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    factor = args.downsample_factor
    width, height = manifest["image_size_wh"]
    small_size = (width // factor, height // factor)
    print(f"Building SurgicalSAM2 on cuda:{args.cuda_device}", flush=True)
    predictor = build_predictor(
        args.paper_repo,
        args.config,
        args.checkpoint,
    )

    report_rows: list[dict[str, Any]] = []
    contact_rows: list[np.ndarray] = []
    with torch.inference_mode(), torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
    ):
        for row, annotation in zip(
            manifest["anchors"],
            prompts["anchors"],
            strict=True,
        ):
            strict_slot = int(row["strict_pair_slot"])
            side_panels = []
            side_report: dict[str, Any] = {}
            for side in ("left", "right"):
                image_path = args.root / row[f"{side}_image"]
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(image_path)
                small = cv2.resize(
                    image,
                    small_size,
                    interpolation=cv2.INTER_AREA,
                )
                predictor.frame_idx = 0
                predictor.load_first_frame(small)
                logits = None
                for obj_id, object_name in enumerate(("shaft", "distal")):
                    points, labels = object_prompts(
                        annotation["views"][side],
                        object_name=object_name,
                        downsample_factor=factor,
                    )
                    _, _, logits = predictor.add_new_points(
                        frame_idx=0,
                        obj_id=obj_id,
                        points=points,
                        labels=labels,
                    )
                assert logits is not None
                masks_small = unpack_logits(logits)
                if len(masks_small) != 2:
                    raise RuntimeError(
                        f"Expected two objects, received {len(masks_small)}"
                    )
                masks_full = [
                    cv2.resize(
                        mask.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    for mask in masks_small
                ]
                shaft, distal = masks_full
                overlay = tint_parts(image, shaft, distal)
                prompt_view = draw_prompts(image, annotation["views"][side])
                output = np.concatenate(
                    [
                        add_title(prompt_view, f"slot {strict_slot:04d} {side} prompts"),
                        add_title(
                            overlay,
                            (
                                "blue=shaft  orange=distal  "
                                "magenta=overlap"
                            ),
                        ),
                    ],
                    axis=1,
                )
                scale = 920.0 / output.shape[1]
                side_panels.append(
                    cv2.resize(
                        output,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_AREA,
                    )
                )
                view = annotation["views"][side]
                side_report[side] = {
                    "shaft": prompt_metrics(
                        shaft,
                        positive=view["shaft_positive"],
                        negative=[
                            *view["shaft_negative"],
                            *view["distal_positive"],
                        ],
                    ),
                    "distal": prompt_metrics(
                        distal,
                        positive=view["distal_positive"],
                        negative=[
                            *view["distal_negative"],
                            *view["shaft_positive"],
                        ],
                    ),
                    "overlap_area_px": int((shaft & distal).sum()),
                    "union_area_px": int((shaft | distal).sum()),
                }
                individual = (
                    args.output_dir
                    / f"slot_{strict_slot:04d}_{side}_two_object_overlay.png"
                )
                if not cv2.imwrite(str(individual), overlay):
                    raise RuntimeError(f"Failed to write {individual}")
            contact_rows.append(np.concatenate(side_panels, axis=0))
            report_rows.append(
                {
                    "strict_pair_slot": strict_slot,
                    "views": side_report,
                }
            )
            print(f"Finished prompt anchor {strict_slot}", flush=True)

    contact_sheet = args.output_dir / "multianchor_sam2_overview.png"
    if not cv2.imwrite(
        str(contact_sheet),
        np.concatenate(contact_rows, axis=0),
    ):
        raise RuntimeError(f"Failed to write {contact_sheet}")
    report = {
        "schema": "super_paper_lnd_multianchor_sam2_prompt_test_v2",
        "passed": True,
        "legend_bgr": {
            "shaft": [255, 120, 20],
            "distal": [30, 190, 255],
            "overlap": [220, 40, 220],
        },
        "anchors": report_rows,
        "outputs": {"contact_sheet": str(contact_sheet)},
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Contact sheet: {contact_sheet}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
