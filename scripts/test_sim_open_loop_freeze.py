#!/usr/bin/env python3
"""验证 80/20 分界会取消未决刚度候选且保持已提交材料。"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import torch

from example_embodied_super_offline import SuperPlaybackControls


class FakeUpdater:
    def __init__(self, candidate: object) -> None:
        self.pending_candidate = candidate
        self.reject_reason: str | None = None

    def reject(self, reason: str, candidate: object) -> dict[str, object]:
        assert candidate is self.pending_candidate
        self.reject_reason = reason
        self.pending_candidate = None
        return {"status": "rejected", "rejection_reason": reason}


def main() -> None:
    candidate = object()
    pending = SimpleNamespace(
        candidate=candidate,
        frame_index=239,
        commands=[object()],
    )
    controls = object.__new__(SuperPlaybackControls)
    controls.current_frame_index = 240
    controls.current_timestep = 8.0
    controls._current_action_phase = "hold"
    controls._evaluation_open_loop_start_frame = 240
    controls._evaluation_open_loop_entered = False
    controls._pending_stiffness_validation = pending
    controls.stiffness_updater = FakeUpdater(candidate)
    controls.stiffness_metrics_recorder = None
    controls._last_stiffness_validation_metrics = None
    controls._active_stiffness_evaluations = [object()]
    controls._stiffness_history = deque((object(),))
    controls._previous_visual_residual = torch.ones(2)

    controls._enter_evaluation_open_loop_if_needed()

    assert controls._evaluation_open_loop_entered
    assert controls._pending_stiffness_validation is None
    assert controls.stiffness_updater.pending_candidate is None
    assert controls.stiffness_updater.reject_reason == "future_open_loop_freeze"
    assert controls._last_stiffness_validation_metrics[
        "validation_status"
    ] == "cancelled_at_future_split"
    assert not controls._active_stiffness_evaluations
    assert not controls._stiffness_history
    assert controls._previous_visual_residual is None

    # 重复进入不能再次修改或拒绝已冻结状态。
    controls._enter_evaluation_open_loop_if_needed()
    assert controls.stiffness_updater.reject_reason == "future_open_loop_freeze"
    print("sim open-loop freeze gate: PASS")


if __name__ == "__main__":
    main()
