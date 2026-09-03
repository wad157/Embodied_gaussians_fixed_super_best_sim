#!/usr/bin/env python3
"""Fuse selected SUPER stereo depths into an unsmoothed table-frame surface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "data/super/grasp5_native"
MULTIVIEW_ROOT = NATIVE_ROOT / "tissue_multiview_v1"
V9_ROOT = NATIVE_ROOT / "bodies_v9_dense_0p5mm_rigid_tissue"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse selected stereo tissue depths without spatial smoothing."
    )
    parser.add_argument(
        "--stage-b-report",
        type=Path,
        default=MULTIVIEW_ROOT / "stage_b_report.json",
    )
    parser.add_argument(
        "--stage-c-report",
        type=Path,
        default=MULTIVIEW_ROOT / "stage_c_depth_report.json",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=NATIVE_ROOT / "calib_rectified.json",
    )
    parser.add_argument(
        "--table-frame",
        type=Path,
        default=REPO_ROOT / "data/super/table_frame.json",
    )
    parser.add_argument(
        "--v9-tissue",
        type=Path,
        default=V9_ROOT / "tissue.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MULTIVIEW_ROOT / "rest_surface_v1",
    )
    parser.add_argument("--surface-spacing-mm", type=float, default=0.82)
    parser.add_argument("--minimum-height-mm", type=float, default=2.5)
    parser.add_argument("--maximum-height-mm", type=float, default=15.5)
    parser.add_argument("--xy-margin-mm", type=float, default=1.0)
    parser.add_argument("--maximum-hole-diameter-mm", type=float, default=10.0)
    parser.add_argument("--maximum-dual-disagreement-mm", type=float, default=1.5)
    parser.add_argument(
        "--confidence-policy",
        choices=("acceptable", "dense_union"),
        default="acceptable",
        help=(
            "acceptable reproduces the audited stereo-confidence gate; "
            "dense_union retains every finite FoundationStereo depth inside "
            "the per-view tissue visibility mask and uses confidence only as audit."
        ),
    )
    parser.add_argument(
        "--retain-inconsistent-dual",
        action="store_true",
        help=(
            "Keep cells seen by both sides even when their depths disagree; "
            "the disagreement remains in the report instead of deleting the cell."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def five_number(values: np.ndarray) -> list[float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    return np.percentile(finite, [0, 5, 50, 95, 100]).tolist()


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def selected_frames(stage_b: dict, side: str) -> list[int]:
    return [
        int(item[f"{side}_frame"])
        for item in stage_b["coverage"]["selected_temporal_order"]
    ]


def v9_table_points(v9_tissue: dict) -> np.ndarray:
    transform = np.asarray(v9_tissue["X_WB"], dtype=np.float64)
    points = np.asarray(v9_tissue["particles"]["means"], dtype=np.float64)
    return transform_points(points, transform)


def robust_cell_medians(
    cell_ids: np.ndarray,
    heights: np.ndarray,
    cell_count: int,
) -> np.ndarray:
    order = np.argsort(cell_ids, kind="stable")
    ids = cell_ids[order]
    values = heights[order]
    unique, starts, counts = np.unique(
        ids, return_index=True, return_counts=True
    )
    output = np.full(cell_count, np.nan, dtype=np.float32)
    for cell_id, start, count in zip(unique, starts, counts, strict=True):
        output[int(cell_id)] = np.median(values[start : start + count])
    return output


def fill_small_holes(
    valid: np.ndarray,
    heights: np.ndarray,
    maximum_diameter_cells: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill enclosed observation gaps without modifying any observed height.

    The unknown cells solve a discrete Laplace equation with the observed
    boundary fixed.  This is topology completion, not smoothing: values in
    ``valid`` are copied bit-for-bit to the output.
    """
    filled_envelope = ndimage.binary_fill_holes(valid)
    holes = filled_envelope & ~valid
    labels, count = ndimage.label(holes)
    accepted = np.zeros_like(valid)
    for label_id in range(1, count + 1):
        region = labels == label_id
        rows, columns = np.nonzero(region)
        if not len(rows):
            continue
        diameter = max(
            int(rows.max() - rows.min() + 1),
            int(columns.max() - columns.min() + 1),
        )
        if diameter <= maximum_diameter_cells:
            accepted |= region
    if not np.any(accepted):
        return valid.copy(), heights.copy(), accepted

    output_heights = heights.copy()
    unknown_rows, unknown_columns = np.nonzero(accepted)
    unknown_ids = np.full(valid.shape, -1, dtype=np.int32)
    unknown_ids[unknown_rows, unknown_columns] = np.arange(
        len(unknown_rows), dtype=np.int32
    )
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    rhs = np.zeros(len(unknown_rows), dtype=np.float64)
    for equation, (row, column) in enumerate(
        zip(unknown_rows, unknown_columns, strict=True)
    ):
        neighbor_count = 0
        for drow, dcolumn in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor_row = int(row + drow)
            neighbor_column = int(column + dcolumn)
            if not (
                0 <= neighbor_row < valid.shape[0]
                and 0 <= neighbor_column < valid.shape[1]
            ):
                continue
            if accepted[neighbor_row, neighbor_column]:
                matrix_rows.append(equation)
                matrix_columns.append(
                    int(unknown_ids[neighbor_row, neighbor_column])
                )
                matrix_values.append(-1.0)
                neighbor_count += 1
            elif valid[neighbor_row, neighbor_column]:
                rhs[equation] += float(
                    heights[neighbor_row, neighbor_column]
                )
                neighbor_count += 1
        if not neighbor_count:
            raise RuntimeError("Accepted hole cell has no 4-neighborhood")
        matrix_rows.append(equation)
        matrix_columns.append(equation)
        matrix_values.append(float(neighbor_count))
    laplacian = csr_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(unknown_rows), len(unknown_rows)),
    )
    solved = spsolve(laplacian, rhs)
    if not np.isfinite(solved).all():
        raise RuntimeError("Topology hole completion produced non-finite height")
    output_heights[unknown_rows, unknown_columns] = solved
    if not np.array_equal(
        output_heights[valid].view(np.uint32),
        heights[valid].view(np.uint32),
    ):
        raise RuntimeError("Topology completion modified an observed height")
    return valid | accepted, output_heights, accepted


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    labels, count = ndimage.label(mask)
    if count == 0:
        raise RuntimeError("Fused rest surface contains no occupied cell")
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest_id = int(np.argmax(component_sizes))
    largest = labels == largest_id
    removed = int(mask.sum() - largest.sum())
    return largest, count, removed


def build_top_mesh(
    *,
    occupied: np.ndarray,
    heights: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    source_class: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertex_ids = np.full(occupied.shape, -1, dtype=np.int32)
    rows, columns = np.nonzero(occupied)
    vertex_ids[rows, columns] = np.arange(len(rows), dtype=np.int32)
    vertices = np.stack(
        (x_centers[rows], y_centers[columns], heights[rows, columns]),
        axis=1,
    )
    vertex_classes = source_class[rows, columns]
    faces: list[tuple[int, int, int]] = []
    for row in range(occupied.shape[0] - 1):
        for column in range(occupied.shape[1] - 1):
            corners = (
                (row, column),
                (row + 1, column),
                (row + 1, column + 1),
                (row, column + 1),
            )
            if not all(occupied[item] for item in corners):
                continue
            ids = [int(vertex_ids[item]) for item in corners]
            z = [float(heights[item]) for item in corners]
            if abs(z[0] - z[2]) <= abs(z[1] - z[3]):
                faces.extend(((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])))
            else:
                faces.extend(((ids[0], ids[1], ids[3]), (ids[1], ids[2], ids[3])))
    return (
        vertices.astype(np.float32),
        np.asarray(faces, dtype=np.int32),
        vertex_classes.astype(np.uint8),
        vertex_ids,
    )


def write_surface_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    source_class: np.ndarray,
) -> None:
    colors = np.zeros((len(vertices), 3), dtype=np.float64)
    colors[source_class == 1] = (0.20, 0.55, 0.95)  # left-only
    colors[source_class == 2] = (0.95, 0.55, 0.20)  # right-only
    colors[source_class == 3] = (0.20, 0.85, 0.35)  # dual
    colors[source_class == 4] = (0.90, 0.20, 0.75)  # inferred
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False):
        raise RuntimeError(f"Failed to write {path}")


def neighbor_jumps(occupied: np.ndarray, heights: np.ndarray) -> np.ndarray:
    horizontal = (
        occupied[:-1, :] & occupied[1:, :]
    )
    vertical = (
        occupied[:, :-1] & occupied[:, 1:]
    )
    return np.concatenate(
        (
            np.abs(heights[:-1, :][horizontal] - heights[1:, :][horizontal]),
            np.abs(heights[:, :-1][vertical] - heights[:, 1:][vertical]),
        )
    )


def main() -> None:
    args = parse_args()
    if args.surface_spacing_mm <= 0.0:
        raise ValueError("surface-spacing-mm must be positive")
    outputs = {
        "asset": args.output_dir / "rest_surface.npz",
        "ply": args.output_dir / "rest_surface.ply",
        "report": args.output_dir / "report.json",
    }
    collisions = [path for path in outputs.values() if path.exists()]
    if collisions and not args.overwrite:
        raise FileExistsError(
            "Surface outputs already exist; pass --overwrite:\n- "
            + "\n- ".join(str(path) for path in collisions)
        )

    stage_b = read_json(args.stage_b_report)
    stage_c = read_json(args.stage_c_report)
    if stage_b.get("status") != "passed_for_stage_c":
        raise RuntimeError("Stage-B report did not pass")
    if not stage_c.get("passed", False):
        raise RuntimeError("Stage-C report did not pass")
    calibration = read_json(args.calibration)
    table_frame = read_json(args.table_frame)
    v9_tissue = read_json(args.v9_tissue)
    K = {
        side: np.asarray(calibration[f"K_{side}_rect"], dtype=np.float64)
        for side in ("left", "right")
    }
    baseline = float(calibration["baseline_m"])
    X_table_left = np.asarray(
        table_frame["X_table_camera"], dtype=np.float64
    )

    spacing = args.surface_spacing_mm / 1000.0
    v9_points = v9_table_points(v9_tissue)
    margin = args.xy_margin_mm / 1000.0
    xy_min = v9_points[:, :2].min(axis=0) - margin
    xy_max = v9_points[:, :2].max(axis=0) + margin
    xy_min = np.floor(xy_min / spacing) * spacing
    xy_max = np.ceil(xy_max / spacing) * spacing
    x_centers = np.arange(
        xy_min[0], xy_max[0] + spacing * 0.5, spacing, dtype=np.float64
    )
    y_centers = np.arange(
        xy_min[1], xy_max[1] + spacing * 0.5, spacing, dtype=np.float64
    )
    grid_shape = (len(x_centers), len(y_centers))
    cell_count = int(np.prod(grid_shape))
    observation_heights: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    observation_reports = []

    minimum_height = args.minimum_height_mm / 1000.0
    maximum_height = args.maximum_height_mm / 1000.0
    for side in ("left", "right"):
        frames = selected_frames(stage_b, side)
        for frame in frames:
            depth_path = (
                MULTIVIEW_ROOT / f"depth_{side}/{frame:06d}-depth.npy"
            )
            confidence_path = (
                MULTIVIEW_ROOT
                / f"depth_{side}/{frame:06d}-confidence.npz"
            )
            visibility_path = (
                MULTIVIEW_ROOT
                / f"masks_{side}/stage_b_visibility_proxy/"
                f"{frame:06d}-visibility-proxy.png"
            )
            depth = np.load(depth_path).astype(np.float64)
            with np.load(confidence_path) as loaded:
                acceptable = loaded["stage_c_acceptable"].astype(bool)
                dense_valid = loaded["foundation_dense_valid"].astype(bool)
            visibility_image = cv2.imread(
                str(visibility_path), cv2.IMREAD_GRAYSCALE
            )
            if visibility_image is None:
                raise FileNotFoundError(visibility_path)
            confidence_selected = (
                dense_valid
                if args.confidence_policy == "dense_union"
                else acceptable
            )
            selected = (
                confidence_selected
                & (visibility_image > 0)
                & np.isfinite(depth)
                & (depth > 0.0)
            )
            rows, columns = np.nonzero(selected)
            z = depth[rows, columns]
            points = np.stack(
                (
                    (columns - K[side][0, 2]) * z / K[side][0, 0],
                    (rows - K[side][1, 2]) * z / K[side][1, 1],
                    z,
                ),
                axis=1,
            )
            if side == "right":
                points[:, 0] += baseline
            points_table = transform_points(points, X_table_left)
            valid = (
                np.isfinite(points_table).all(axis=1)
                & (points_table[:, 2] >= minimum_height)
                & (points_table[:, 2] <= maximum_height)
                & (points_table[:, 0] >= xy_min[0] - spacing * 0.5)
                & (points_table[:, 0] <= xy_max[0] + spacing * 0.5)
                & (points_table[:, 1] >= xy_min[1] - spacing * 0.5)
                & (points_table[:, 1] <= xy_max[1] + spacing * 0.5)
            )
            points_table = points_table[valid]
            ix = np.rint(
                (points_table[:, 0] - x_centers[0]) / spacing
            ).astype(np.int32)
            iy = np.rint(
                (points_table[:, 1] - y_centers[0]) / spacing
            ).astype(np.int32)
            inside = (
                (ix >= 0)
                & (ix < grid_shape[0])
                & (iy >= 0)
                & (iy < grid_shape[1])
            )
            ix = ix[inside]
            iy = iy[inside]
            heights = points_table[inside, 2]
            cell_ids = ix.astype(np.int64) * grid_shape[1] + iy
            cell_medians = robust_cell_medians(
                cell_ids, heights, cell_count
            ).reshape(grid_shape)
            observation_heights[side].append(cell_medians)
            observation_reports.append(
                {
                    "side": side,
                    "frame": frame,
                    "selected_pixels": int(selected.sum()),
                    "acceptable_pixels": int(
                        np.count_nonzero(acceptable & (visibility_image > 0))
                    ),
                    "dense_union_only_pixels": int(
                        np.count_nonzero(
                            dense_valid
                            & (visibility_image > 0)
                            & ~acceptable
                        )
                    ),
                    "accepted_height_points": int(len(heights)),
                    "occupied_cells": int(np.isfinite(cell_medians).sum()),
                    "height_mm_min_p05_p50_p95_max": five_number(
                        heights * 1000.0
                    ),
                }
            )

    side_stack = {
        side: np.stack(observation_heights[side], axis=0)
        for side in ("left", "right")
    }
    side_height = {
        side: np.nanmedian(side_stack[side], axis=0)
        for side in ("left", "right")
    }
    side_count = {
        side: np.isfinite(side_stack[side]).sum(axis=0).astype(np.uint8)
        for side in ("left", "right")
    }
    left_valid = np.isfinite(side_height["left"])
    right_valid = np.isfinite(side_height["right"])
    dual = left_valid & right_valid
    dual_disagreement = np.abs(
        side_height["left"] - side_height["right"]
    )
    consistent_dual = dual & (
        dual_disagreement
        <= args.maximum_dual_disagreement_mm / 1000.0
    )
    rejected_dual = dual & ~consistent_dual

    fused_height = np.full(grid_shape, np.nan, dtype=np.float32)
    retained_dual = dual if args.retain_inconsistent_dual else consistent_dual
    fused_height[retained_dual] = (
        0.5
        * (
            side_height["left"][retained_dual]
            + side_height["right"][retained_dual]
        )
    )
    left_only = left_valid & ~right_valid
    right_only = right_valid & ~left_valid
    fused_height[left_only] = side_height["left"][left_only]
    fused_height[right_only] = side_height["right"][right_only]
    observed = np.isfinite(fused_height)
    largest, component_count, removed_cells = largest_component(observed)
    fused_height[~largest] = np.nan
    observed = largest

    maximum_hole_cells = max(
        1.0, args.maximum_hole_diameter_mm / args.surface_spacing_mm
    )
    observed_heights_before_completion = fused_height[observed].copy()
    occupied, fused_height, inferred = fill_small_holes(
        observed,
        fused_height,
        maximum_hole_cells,
    )
    source_class = np.zeros(grid_shape, dtype=np.uint8)
    source_class[occupied & left_only] = 1
    source_class[occupied & right_only] = 2
    source_class[occupied & retained_dual] = 3
    source_class[inferred] = 4

    vertices, faces, vertex_classes, vertex_ids = build_top_mesh(
        occupied=occupied,
        heights=fused_height,
        x_centers=x_centers,
        y_centers=y_centers,
        source_class=source_class,
    )
    if not len(faces):
        raise RuntimeError("Fused rest surface contains no triangles")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outputs["asset"],
        x_centers=x_centers.astype(np.float32),
        y_centers=y_centers.astype(np.float32),
        surface_spacing_m=np.float32(spacing),
        height_grid=fused_height.astype(np.float32),
        occupied_grid=occupied,
        observed_grid=observed,
        inferred_grid=inferred,
        source_class_grid=source_class,
        left_height_grid=side_height["left"].astype(np.float32),
        right_height_grid=side_height["right"].astype(np.float32),
        left_observation_count=side_count["left"],
        right_observation_count=side_count["right"],
        surface_vertices_table=vertices,
        surface_faces=faces,
        surface_vertex_source_class=vertex_classes,
        surface_vertex_ids_grid=vertex_ids,
    )
    write_surface_ply(outputs["ply"], vertices, faces, vertex_classes)

    jumps = neighbor_jumps(occupied, fused_height)
    occupied_count = int(occupied.sum())
    class_counts = {
        "left_only": int(np.count_nonzero(source_class == 1)),
        "right_only": int(np.count_nonzero(source_class == 2)),
        "dual": int(np.count_nonzero(source_class == 3)),
        "inferred": int(np.count_nonzero(source_class == 4)),
    }
    report = {
        "stage": "v10_stage_d_unsmoothed_multiview_rest_surface",
        "runtime_asset_modified": False,
        "parameters": {
            "surface_spacing_mm": args.surface_spacing_mm,
            "minimum_height_mm": args.minimum_height_mm,
            "maximum_height_mm": args.maximum_height_mm,
            "maximum_hole_diameter_mm": args.maximum_hole_diameter_mm,
            "maximum_dual_disagreement_mm": args.maximum_dual_disagreement_mm,
            "confidence_policy": args.confidence_policy,
            "retain_inconsistent_dual": args.retain_inconsistent_dual,
            "spatial_smoothing": False,
            "observed_heights_modified_by_hole_completion": False,
            "hole_completion": (
                "discrete harmonic interpolation on enclosed unobserved "
                "cells only; all observed grid heights remain bit-identical"
            ),
            "fusion": (
                "per-frame per-cell median, per-side temporal median, "
                "then 50/50 left-right mean for retained dual-view cells; "
                "left-only and right-only cells are retained independently"
            ),
        },
        "counts": {
            "grid_shape": list(grid_shape),
            "occupied_cells": occupied_count,
            "surface_vertices": int(len(vertices)),
            "surface_triangles": int(len(faces)),
            "components_before_cleanup": int(component_count),
            "nonlargest_cells_removed": int(removed_cells),
            "dual_disagreement_cells_rejected": int(rejected_dual.sum()),
            "dual_disagreement_cells_retained": int(
                np.count_nonzero(rejected_dual & retained_dual)
            ),
            **class_counts,
        },
        "fractions": {
            key: value / max(occupied_count, 1)
            for key, value in class_counts.items()
        },
        "geometry": {
            "bbox_min_m": vertices.min(axis=0).tolist(),
            "bbox_max_m": vertices.max(axis=0).tolist(),
            "height_mm_min_p05_p50_p95_max": five_number(
                vertices[:, 2] * 1000.0
            ),
            "neighbor_height_jump_mm_min_p05_p50_p95_max": five_number(
                jumps * 1000.0
            ),
            "dual_view_height_difference_mm_min_p05_p50_p95_max": five_number(
                dual_disagreement[dual] * 1000.0
            ),
            "accepted_dual_height_difference_mm_min_p05_p50_p95_max": five_number(
                dual_disagreement[consistent_dual] * 1000.0
            ),
        },
        "observations": observation_reports,
        "inputs": {
            "stage_b_report": {
                "path": str(args.stage_b_report.resolve()),
                "sha256": sha256(args.stage_b_report),
            },
            "stage_c_report": {
                "path": str(args.stage_c_report.resolve()),
                "sha256": sha256(args.stage_c_report),
            },
            "calibration": {
                "path": str(args.calibration.resolve()),
                "sha256": sha256(args.calibration),
            },
            "table_frame": {
                "path": str(args.table_frame.resolve()),
                "sha256": sha256(args.table_frame),
            },
            "v9_tissue_bounds_only": {
                "path": str(args.v9_tissue.resolve()),
                "sha256": sha256(args.v9_tissue),
            },
        },
        "outputs": {
            "asset": str(outputs["asset"].resolve()),
            "ply": str(outputs["ply"].resolve()),
        },
    }
    gates = {
        "stage_c_passed": bool(stage_c.get("passed", False)),
        "single_connected_surface_component": component_count >= 1
        and removed_cells / max(int(observed.sum()) + removed_cells, 1) <= 0.01,
        "inferred_fraction_below_2_percent": (
            class_counts["inferred"] / max(occupied_count, 1) <= 0.02
        ),
        "surface_triangle_budget": len(faces) <= 30000,
        "surface_vertex_budget": len(vertices) <= 20000,
        "finite_vertices": bool(np.isfinite(vertices).all()),
        "height_inside_configured_bounds": bool(
            np.all(vertices[:, 2] >= minimum_height - 1.0e-9)
            and np.all(vertices[:, 2] <= maximum_height + 1.0e-9)
        ),
        "no_spatial_smoothing": True,
        "observed_heights_bit_identical_after_topology_completion": bool(
            np.array_equal(
                fused_height[observed].view(np.uint32),
                observed_heights_before_completion.view(np.uint32),
            )
        ),
    }
    report["gates"] = gates
    report["passed"] = bool(all(gates.values()))
    outputs["report"].write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Unsmoothed multiview rest-surface gates failed")


if __name__ == "__main__":
    main()
