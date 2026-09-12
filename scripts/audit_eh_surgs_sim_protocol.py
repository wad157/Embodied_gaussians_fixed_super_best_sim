#!/usr/bin/env python3
"""Fail-closed EH-SurGS audit layered on the established SIM protocol audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import audit_endogaussian_sim_protocol as shared_audit  # noqa: E402


EH_SURGS_COMMIT = "73fa04e6f5c21cc1685f728eccb1332e81ce620c"
CAMERA_PATCH_SHA256 = "f75e5d151e7f885024e6c75ccc3a1457ebc86389d8277d433cae58175bc310d8"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=("sim01", "sim02", "sim03"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--baseline-root", type=Path, default=REPO_ROOT / "baselines" / "EH-SurGS"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Reuse the established dataset/split/depth/mask/query/calibration checks in full.
    shared_args = argparse.Namespace(
        dataset_key=args.dataset_key,
        dataset=args.dataset,
        baseline_root=REPO_ROOT / "baselines" / "EndoGaussian",
        output=None,
    )
    report = shared_audit.audit(shared_args)

    baseline = args.baseline_root.expanduser().resolve()
    commit = git(baseline, "rev-parse", "HEAD").strip()
    if commit != EH_SURGS_COMMIT:
        raise ValueError("EH-SurGS commit 错误：{} != {}".format(commit, EH_SURGS_COMMIT))
    changed = [
        path
        for path in git(baseline, "diff", "--name-only", "HEAD").splitlines()
        if "/__pycache__/" not in path and not path.endswith(".pyc")
    ]
    if changed != ["train.py"]:
        raise ValueError("EH-SurGS checkout 存在未批准改动：{}".format(changed))
    patch = git(baseline, "diff", "--", "train.py").encode("utf-8")
    patch_hash = hashlib.sha256(patch).hexdigest()
    if patch_hash != CAMERA_PATCH_SHA256:
        raise ValueError("相机兼容补丁哈希错误：{}".format(patch_hash))

    report["schema"] = "fixedsuperbest.eh_surgs_protocol_audit.v1"
    report.pop("endogaussian", None)
    report["eh_surgs"] = {
        "root": str(baseline),
        "commit": commit,
        "algorithm_changes": [
            "train.py reads focal length and principal point from the active SIM camera for adaptive-motion block assignment"
        ],
        "camera_compatibility_patch_sha256": patch_hash,
        "all_other_tracked_sources_clean": True,
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("拒绝覆盖协议审计结果：{}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
