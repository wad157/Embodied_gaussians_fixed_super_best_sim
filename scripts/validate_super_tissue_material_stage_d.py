#!/usr/bin/env python3
"""Independent artifact audit for SUPER tissue calibration stage D."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPO_ROOT
    / "data/super/tissue_calibration_v1/stage_d_material_calibration"
)
EXPECTED_SPLITS = {
    "calibration": [270, 906],
    "validation": [907, 1276],
    "test": [1277, 1439],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    report_path = args.report or args.root / "stage_d_calibration_report.json"
    output_path = (
        args.output or args.root / "stage_d_artifact_validation.json"
    )
    material_path = args.root / "calibrated_material.json"
    candidate_root = args.root / "candidate_reports"

    report = read_json(report_path)
    material = read_json(material_path)
    candidate_paths = sorted(candidate_root.glob("*.json"))
    candidates = {path.stem: read_json(path) for path in candidate_paths}
    verify_a = candidates.get("verify_00", {})
    verify_b = candidates.get("verify_01", {})

    input_hashes_valid = True
    input_hash_records: dict[str, Any] = {}
    for name, record in verify_a.get("inputs", {}).items():
        path = Path(record["path"])
        actual = sha256(path) if path.is_file() else None
        matches = actual == record["sha256"]
        input_hashes_valid = input_hashes_valid and matches
        input_hash_records[name] = {
            "path": str(path),
            "expected_sha256": record["sha256"],
            "actual_sha256": actual,
            "matches": matches,
        }

    search_candidates = [
        candidate
        for name, candidate in candidates.items()
        if name.startswith(("coarse_", "local_"))
    ]
    ranked_search = sorted(
        search_candidates,
        key=lambda candidate: candidate["objective"][
            "calibration_objective_mm"
        ],
    )
    selected_pair = (
        float(material["young_modulus_pa"]),
        float(material["velocity_damping_per_second"]),
    )
    best_pair = (
        float(ranked_search[0]["candidate"]["young_modulus_pa"]),
        float(
            ranked_search[0]["candidate"][
                "velocity_damping_per_second"
            ]
        ),
    )

    split_frames_valid = True
    split_frames_disjoint = True
    seen_frames: set[int] = set()
    for name, (start, end) in EXPECTED_SPLITS.items():
        frames = verify_a.get("split_metrics", {}).get(name, {}).get(
            "evaluated_frames", []
        )
        split_frames_valid = split_frames_valid and bool(frames) and all(
            start <= int(frame) <= end for frame in frames
        )
        frame_set = {int(frame) for frame in frames}
        split_frames_disjoint = split_frames_disjoint and not (
            seen_frames & frame_set
        )
        seen_frames.update(frame_set)

    frozen_parameters_valid = all(
        candidate.get("fixed_parameters", {}).get("poisson_ratio") == 0.0
        and candidate.get("fixed_parameters", {}).get("gravity_m_s2") == 0.0
        and candidate.get("fixed_parameters", {}).get(
            "visual_force_iterations"
        )
        == 0
        and not candidate.get("fixed_parameters", {}).get(
            "residual_mapping_enabled", True
        )
        and not candidate.get("fixed_parameters", {}).get(
            "online_stiffness_optimization_enabled", True
        )
        for candidate in candidates.values()
    )
    all_candidates_physical = all(
        candidate.get("passed", False)
        and candidate.get("stability", {}).get("finite", False)
        and candidate.get("stability", {}).get(
            "inverted_tetrahedra_max", 1
        )
        == 0
        and candidate.get("stability", {}).get(
            "minimum_tetrahedron_volume_ratio", 0.0
        )
        > 0.0
        and candidate.get("stability", {}).get(
            "maximum_anchor_drift_m", 1.0
        )
        == 0.0
        for candidate in candidates.values()
    )
    verify_pair_matches = (
        verify_a.get("candidate") == verify_b.get("candidate")
        and verify_a.get("objective") == verify_b.get("objective")
        and verify_a.get("split_metrics") == verify_b.get("split_metrics")
        and verify_a.get("state_hashes") == verify_b.get("state_hashes")
    )
    dense_valid = all(
        verify.get("stability", {}).get("checked_frame_count") == 1440
        and verify.get("stability", {}).get("finite", False)
        and verify.get("stability", {}).get(
            "inverted_tetrahedra_max", 1
        )
        == 0
        and verify.get("stability", {}).get(
            "minimum_tetrahedron_volume_ratio", 0.0
        )
        > 0.0
        for verify in (verify_a, verify_b)
    )

    gates = {
        "report_schema_and_pass": bool(
            report.get("schema")
            == "super_tissue_stage_d_material_calibration_report_v1"
            and report.get("passed", False)
        ),
        "material_schema_and_pass": bool(
            material.get("schema") == "super_tissue_calibrated_material_v1"
            and material.get("passed", False)
        ),
        "expected_continuous_splits": bool(
            report.get("continuous_time_splits_inclusive")
            == EXPECTED_SPLITS
        ),
        "split_evaluation_frames_in_bounds": split_frames_valid,
        "split_evaluation_frames_disjoint": split_frames_disjoint,
        "future_intervals_not_used_for_selection": bool(
            "validation and test remain future-only"
            in report.get("method", {}).get("selection_rule", "")
        ),
        "expected_candidate_counts": bool(
            report.get("candidate_counts")
            == {
                "coarse": 13,
                "local": 8,
                "full_sequence_finalists": 3,
                "dense_selected_verification": 2,
                "total_replays": 26,
            }
            and len(candidate_paths) == 26
        ),
        "all_candidate_reports_pass_physics": all_candidates_physical,
        "only_two_material_scalars_varied": frozen_parameters_valid,
        "selected_is_calibration_search_minimum": bool(
            ranked_search and selected_pair == best_pair
        ),
        "selected_material_matches_dense_verification": bool(
            selected_pair
            == (
                verify_a.get("candidate", {}).get("young_modulus_pa"),
                verify_a.get("candidate", {}).get(
                    "velocity_damping_per_second"
                ),
            )
        ),
        "selected_not_worse_than_frozen_baseline": bool(
            report.get("calibration_improvement_fraction_vs_baseline", -1.0)
            >= 0.0
        ),
        "verification_pair_exact": verify_pair_matches,
        "dense_1440_frame_physics_valid": dense_valid,
        "report_determinism_flags_pass": bool(
            report.get("determinism", {}).get(
                "exact_calibration_objective", False
            )
            and report.get("determinism", {}).get(
                "exact_landmark_and_final_particle_hashes", False
            )
        ),
        "input_artifact_hashes_match": input_hashes_valid,
    }
    passed = all(gates.values())
    validation = {
        "schema": "super_tissue_stage_d_artifact_validation_v1",
        "stage": "D_independent_artifact_validation",
        "passed": passed,
        "gate_count": len(gates),
        "passed_gate_count": sum(bool(value) for value in gates.values()),
        "gates": gates,
        "selected_material": {
            "young_modulus_pa": selected_pair[0],
            "velocity_damping_per_second": selected_pair[1],
        },
        "input_hash_validation": input_hash_records,
        "artifacts": {
            "calibration_report": {
                "path": str(report_path.resolve()),
                "sha256": sha256(report_path),
            },
            "calibrated_material": {
                "path": str(material_path.resolve()),
                "sha256": sha256(material_path),
            },
            "candidate_report_count": len(candidate_paths),
            "candidate_report_sha256": {
                path.name: sha256(path) for path in candidate_paths
            },
            "calibrator": {
                "path": str(
                    (
                        REPO_ROOT
                        / "scripts/calibrate_super_tissue_material_stage_d.py"
                    ).resolve()
                ),
                "sha256": sha256(
                    REPO_ROOT
                    / "scripts/calibrate_super_tissue_material_stage_d.py"
                ),
            },
            "validator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
    }
    output_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Stage-D artifact validation: "
        f"{validation['passed_gate_count']}/{validation['gate_count']} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    if not passed:
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit(f"Failed gates: {failed}")


if __name__ == "__main__":
    main()
