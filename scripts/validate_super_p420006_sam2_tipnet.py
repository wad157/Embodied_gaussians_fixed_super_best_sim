#!/usr/bin/env python3
"""Validate upstream ContourTipNet jaw-tip detections on stereo SAM2 masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
VISUAL_ROOT = (
    REPO_ROOT
    / "data/super/psm_visual_calibration/raw_p420006_stereo_v1"
)
SAM2_ROOT = VISUAL_ROOT / "surgicalsam2_stereo_10anchor_v1"
PAPER_REPO = Path("/Media_HDD/jwshan/wad/online_dvrk_tracking")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-root", type=Path, default=VISUAL_ROOT)
    parser.add_argument(
        "--masks",
        type=Path,
        default=SAM2_ROOT / "stereo_surgicalsam2_masks.npz",
    )
    parser.add_argument("--paper-repo", type=Path, default=PAPER_REPO)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PAPER_REPO / "ContourTipNet/models/cnn_model.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAM2_ROOT / "tipnet_validation",
    )
    parser.add_argument("--cuda-device", type=int, default=1)
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_mask(
    packed: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    return np.unpackbits(
        packed,
        bitorder="little",
        count=shape[0] * shape[1],
    ).reshape(shape).astype(bool)


def assignment_errors(
    detected: np.ndarray,
    manual: np.ndarray,
) -> tuple[np.ndarray, bool]:
    direct = np.linalg.norm(detected - manual, axis=1)
    swapped = np.linalg.norm(detected[::-1] - manual, axis=1)
    if swapped.sum() < direct.sum():
        return swapped, True
    return direct, False


def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    detected: np.ndarray | None,
    manual: np.ndarray,
    title: str,
) -> np.ndarray:
    result = image.copy()
    tint = np.zeros_like(result)
    tint[mask] = (255, 110, 20)
    result[mask] = cv2.addWeighted(
        result[mask],
        0.55,
        tint[mask],
        0.45,
        0.0,
    )
    for index, point in enumerate(manual):
        cv2.drawMarker(
            result,
            tuple(np.rint(point).astype(int)),
            (60, 255, 60),
            cv2.MARKER_CROSS,
            24,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            f"M{index + 1}",
            tuple(np.rint(point + [8, -8]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 255, 60),
            2,
            cv2.LINE_AA,
        )
    if detected is not None:
        for index, point in enumerate(detected):
            center = tuple(np.rint(point).astype(int))
            cv2.circle(result, center, 10, (30, 40, 255), 3)
            cv2.putText(
                result,
                f"T{index + 1}",
                tuple(np.rint(point + [8, 18]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 40, 255),
                2,
                cv2.LINE_AA,
            )
    cv2.rectangle(result, (0, 0), (result.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        result,
        title,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    args = parse_args()
    for path in (
        args.visual_root / "pair_manifest.json",
        args.masks,
        args.checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.cuda_device)
    sys.path.insert(0, str(args.paper_repo))
    from diffcali.utils.contour_tip_net import (
        Tip2DNet,
        detect_keypoints_2d,
    )

    model = Tip2DNet().cuda().eval()
    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=f"cuda:{args.cuda_device}",
            weights_only=True,
        )
    )
    manifest = json.loads(
        (args.visual_root / "pair_manifest.json").read_text(encoding="utf-8")
    )
    with np.load(args.masks, allow_pickle=False) as masks:
        shape = tuple(masks["mask_shape"].astype(int).tolist())
        packed = {
            "left": masks["left_masks_packbits"].copy(),
            "right": masks["right_masks_packbits"].copy(),
        }
    scale = np.asarray(
        [
            shape[1] / manifest["image_size_wh"][0],
            shape[0] / manifest["image_size_wh"][1],
        ],
        dtype=np.float32,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    measurements: list[dict[str, Any]] = []
    panels: list[np.ndarray] = []
    all_errors: list[float] = []
    missing = 0
    for row in manifest["keyframes"]:
        keyframe = int(row["keyframe_index"])
        slot = int(row["strict_pair_slot"])
        annotation = json.loads(
            (
                args.visual_root
                / "annotations"
                / f"keyframe_{keyframe:02d}.json"
            ).read_text(encoding="utf-8")
        )
        side_panels: list[np.ndarray] = []
        measurement: dict[str, Any] = {
            "keyframe_index": keyframe,
            "strict_pair_slot": slot,
            "views": {},
        }
        for side in ("left", "right"):
            mask = unpack_mask(packed[side][slot], shape)
            mask_tensor = torch.as_tensor(
                mask,
                device=f"cuda:{args.cuda_device}",
                dtype=torch.float32,
            )
            with torch.inference_mode():
                result = detect_keypoints_2d(model, mask_tensor)
            detected = (
                result.detach().cpu().numpy()
                if result is not None
                else None
            )
            manual = np.asarray(
                [
                    annotation["views"][side]["tips"]["jaw_1"][
                        "point_xy"
                    ],
                    annotation["views"][side]["tips"]["jaw_2"][
                        "point_xy"
                    ],
                ],
                dtype=np.float32,
            ) * scale
            if detected is None:
                errors = None
                swapped = None
                missing += 1
            else:
                errors_array, swapped = assignment_errors(
                    detected,
                    manual,
                )
                errors = errors_array.tolist()
                all_errors.extend(errors)
            measurement["views"][side] = {
                "detected_xy": (
                    detected.tolist() if detected is not None else None
                ),
                "manual_xy_at_mask_resolution": manual.tolist(),
                "per_tip_assignment_error_px": errors,
                "assignment_swapped": swapped,
            }
            image = cv2.imread(
                str(args.visual_root / row[f"{side}_image"]),
                cv2.IMREAD_COLOR,
            )
            image = cv2.resize(
                image,
                (shape[1], shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            side_panels.append(
                overlay(
                    image,
                    mask,
                    detected,
                    manual,
                    f"KF {keyframe:02d} {side}",
                )
            )
        measurements.append(measurement)
        panels.append(np.concatenate(side_panels, axis=1))
    sheet_path = args.output_dir / "tipnet_vs_manual_contact_sheet.png"
    if not cv2.imwrite(str(sheet_path), np.concatenate(panels, axis=0)):
        raise RuntimeError(f"Failed to write {sheet_path}")
    errors_np = np.asarray(all_errors, dtype=np.float64)
    report = {
        "schema": "super_p420006_stereo_tipnet_validation_v1",
        "model": {
            "upstream_repository": str(args.paper_repo),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
        },
        "inputs": {
            "masks": str(args.masks),
            "masks_sha256": sha256(args.masks),
        },
        "summary": {
            "view_count": 2 * len(manifest["keyframes"]),
            "missing_two_tip_detection_count": missing,
            "tip_assignment_error_px": {
                "count": len(all_errors),
                "median": (
                    float(np.median(errors_np)) if len(errors_np) else None
                ),
                "p95": (
                    float(np.percentile(errors_np, 95))
                    if len(errors_np)
                    else None
                ),
                "maximum": (
                    float(errors_np.max()) if len(errors_np) else None
                ),
            },
        },
        "measurements": measurements,
        "contact_sheet": str(sheet_path),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {report_path}")
    print(f"Contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
