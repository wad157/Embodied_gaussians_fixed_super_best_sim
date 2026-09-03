#!/usr/bin/env python3
"""CPU-only behavioral gate for the contact-limited PSM jaw actuator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

from embodied_environments.super_embodied.psm_lnd_kinematics import (  # noqa: E402
    TissueResistedJawActuator,
)


def main() -> None:
    actuator = TissueResistedJawActuator(
        0.50,
        maximum_closing_speed_rad_s=2.4,
        minimum_contacts_per_jaw=8,
    )
    actuator.set_command(-0.10)
    first = actuator.step(
        1.0 / 60.0,
        contact_counts=(0, 0),
        resistance_m2=(0.0, 0.0),
    )
    rate_limited = np.isclose(first, 0.46)

    unilateral = actuator.step(
        1.0 / 60.0,
        contact_counts=(20, 0),
        resistance_m2=(1.0e-5, 0.0),
    )
    unilateral_does_not_block = (
        np.isclose(unilateral, 0.42)
        and not actuator.closure_blocked_by_tissue
    )

    compliant_angles = []
    for _ in range(3):
        compliant_angles.append(
            actuator.step(
                1.0 / 60.0,
                contact_counts=(20, 16),
                resistance_m2=(1.0e-5, 8.0e-6),
            )
        )
    blocked_angle = compliant_angles[-1]
    bilateral_tissue_blocks = (
        np.allclose(compliant_angles, (0.38, 0.34, 0.30))
        and actuator.closure_blocked_by_tissue
        and actuator.closing_requested
    )
    held_angle = actuator.step(
        1.0 / 60.0,
        contact_counts=(20, 16),
        resistance_m2=(1.0e-5, 8.0e-6),
    )
    block_holds_angle = np.isclose(held_angle, blocked_angle)

    actuator.set_command(0.70)
    opening_releases_immediately = (
        np.isclose(actuator.actual_angle_rad, 0.70)
        and not actuator.closure_blocked_by_tissue
        and not actuator.closing_requested
    )

    gates = {
        "closure_rate_limited_not_teleported": bool(rate_limited),
        "unilateral_contact_does_not_fake_grip": bool(
            unilateral_does_not_block
        ),
        "bilateral_tissue_resistance_blocks_motor": bool(
            bilateral_tissue_blocks
        ),
        "bilateral_contact_has_finite_compliant_travel": bool(
            np.isclose(unilateral - blocked_angle, 0.12)
        ),
        "blocked_motor_holds_actual_angle": bool(block_holds_angle),
        "opening_command_releases_immediately": bool(
            opening_releases_immediately
        ),
    }
    print(
        json.dumps(
            {
                "passed": all(gates.values()),
                "gates": gates,
                "blocked_angle_rad": float(blocked_angle),
                "final_open_angle_rad": float(actuator.actual_angle_rad),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if all(gates.values()) else 1)


if __name__ == "__main__":
    main()
