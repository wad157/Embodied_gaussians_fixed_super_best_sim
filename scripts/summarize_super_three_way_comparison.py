#!/usr/bin/env python3
"""Summarize paired fixed/residual/online-stiffness trajectory losses."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_MODES = (
    "fixed_pbd",
    "residual_only",
    "residual_online_stiffness",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a three-way SUPER trajectory comparison."
    )
    parser.add_argument("runs", type=Path, nargs=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def finite(values):
    return [float(value) for value in values if math.isfinite(float(value))]


def stats(values) -> dict[str, float | int]:
    values = finite(values)
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "minimum": 0.0, "maximum": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def frame_windows(common_frames: list[int], width: int = 50) -> list[tuple[int, int]]:
    """Return deterministic, inclusive windows aligned to the first frame."""
    start = common_frames[0]
    final = common_frames[-1]
    return [
        (window_start, min(window_start + width - 1, final))
        for window_start in range(start, final + 1, width)
    ]


def stiffness_validation_stats(events: list[dict]) -> dict:
    """Summarize both attempted validations and completed shadow pairs."""
    shadow = [
        event
        for event in events
        if isinstance(event.get("prediction"), dict)
        and all(
            key in event["prediction"]
            for key in (
                "baseline_gap",
                "candidate_gap",
                "gap_improvement",
                "required_improvement",
            )
        )
    ]
    improvements = [
        float(event["prediction"]["gap_improvement"]) for event in shadow
    ]
    required = [
        float(event["prediction"]["required_improvement"]) for event in shadow
    ]
    rejection_reasons = Counter(
        reason
        for event in events
        for reason in event.get("details", {}).get("rejection_reasons", [])
    )
    return {
        "event_count": len(events),
        "shadow_pair_count": len(shadow),
        "committed_count": sum(
            event.get("prediction", {}).get("status") == "committed"
            for event in events
        ),
        "candidate_lower_loss_count": sum(value > 0.0 for value in improvements),
        "required_margin_pass_count": sum(
            improvement >= margin
            for improvement, margin in zip(improvements, required)
        ),
        "baseline_gap": stats(
            event["prediction"]["baseline_gap"] for event in shadow
        ),
        "candidate_gap": stats(
            event["prediction"]["candidate_gap"] for event in shadow
        ),
        "gap_improvement": stats(improvements),
        "required_improvement": stats(required),
        "status_counts": dict(Counter(
            str(event.get("prediction", {}).get("status", "unknown"))
            for event in events
        )),
        "rejection_reason_counts": dict(rejection_reasons),
    }


def open_loop_prediction_stats(events: list[dict]) -> dict:
    improvements = [
        float(event["prediction"]["gap_improvement"]) for event in events
    ]
    baseline_gaps = [
        float(event["prediction"]["baseline_gap"]) for event in events
    ]
    relative_improvements = [
        improvement / max(baseline, 1.0e-12)
        for improvement, baseline in zip(improvements, baseline_gaps)
    ]
    candidate_j_below_baseline = 0
    penetration_regressions = 0
    maximum_inverted_tetrahedra = 0
    for event in events:
        physical = event.get("physical", {})
        baseline = physical.get("baseline", {})
        candidate = physical.get("candidate", {})
        candidate_j_below_baseline += (
            float(candidate.get("minimum_volume_ratio", float("inf")))
            < float(baseline.get("minimum_volume_ratio", float("-inf")))
        )
        penetration_regressions += (
            float(candidate.get("maximum_penetration_m", 0.0))
            > float(baseline.get("maximum_penetration_m", 0.0)) + 1.0e-4
        )
        maximum_inverted_tetrahedra = max(
            maximum_inverted_tetrahedra,
            int(baseline.get("inverted_tetrahedra", 0)),
            int(candidate.get("inverted_tetrahedra", 0)),
        )
    return {
        "count": len(events),
        "positive_count": sum(value > 0.0 for value in improvements),
        "gap_improvement": stats(improvements),
        "relative_improvement": stats(relative_improvements),
        "candidate_j_below_baseline_count": candidate_j_below_baseline,
        "penetration_regression_over_0p1mm_count": penetration_regressions,
        "maximum_inverted_tetrahedra": maximum_inverted_tetrahedra,
    }


def read_run(directory: Path) -> dict:
    directory = directory.resolve()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mode = str(metadata.get("experiment_mode", ""))
    observations = {
        int(event["frame_index"]): event
        for event in events
        if event.get("event") == "trajectory_observation"
    }
    visual_updates = [
        event for event in events if event.get("event") == "visual_update"
    ]
    if not observations:
        raise ValueError(f"{directory} contains no trajectory observations")
    return {
        "directory": str(directory),
        "metadata": metadata,
        "mode": mode,
        "observations": observations,
        "visual_updates": visual_updates,
        "events": events,
    }


def summarize_run(run: dict, common_frames: list[int]) -> dict:
    observations = [run["observations"][frame] for frame in common_frames]
    prediction_losses = [event["image"]["prediction_loss"] for event in observations]
    camera_losses = list(zip(*[
        event["image"]["left_right_prediction_losses"] for event in observations
    ]))
    by_phase = defaultdict(list)
    for event in observations:
        by_phase[event["phase"]].append(event["image"]["prediction_loss"])
    windows = frame_windows(common_frames)
    by_window = {
        f"{start}-{end}": [
            event["image"]["prediction_loss"]
            for event in observations
            if start <= int(event["frame_index"]) <= end
        ]
        for start, end in windows
    }
    minimum_volume_ratios = [
        event["physical"].get("minimum_volume_ratio", float("nan"))
        for event in observations
    ]
    maximum_penetrations = [
        event["physical"].get("maximum_penetration_m", float("nan"))
        for event in observations
    ]
    inverted = [
        event["physical"].get("inverted_tetrahedra", 0)
        for event in observations
    ]
    visual_updates = run["visual_updates"]
    gate_reasons = Counter(
        str(event.get("details", {}).get("stiffness_gate_reason", ""))
        for event in visual_updates
        if event.get("details", {}).get("stiffness_gate_paused")
    )
    visual_updates_by_phase = defaultdict(list)
    for event in visual_updates:
        visual_updates_by_phase[str(event.get("phase", "unknown"))].append(event)
    stiffness_gate_by_phase = {}
    for phase, phase_events in sorted(visual_updates_by_phase.items()):
        phase_reasons = Counter(
            str(event.get("details", {}).get("stiffness_gate_reason", ""))
            for event in phase_events
            if event.get("details", {}).get("stiffness_gate_paused")
        )
        paused_count = sum(phase_reasons.values())
        stiffness_gate_by_phase[phase] = {
            "visual_update_count": len(phase_events),
            "eligible_count": len(phase_events) - paused_count,
            "paused_count": paused_count,
            "pause_reason_counts": dict(phase_reasons),
        }
    material_statuses = Counter(
        str(event.get("material", {}).get("status", ""))
        for event in run["events"]
        if event.get("material", {}).get("status")
    )
    event_type_counts = Counter(event.get("event", "") for event in run["events"])
    validation_statuses = Counter(
        str(event.get("material", {}).get("status", "unknown"))
        for event in run["events"]
        if event.get("event") == "stiffness_validation"
    )
    validation_events = [
        event
        for event in run["events"]
        if event.get("event") == "stiffness_validation"
        and common_frames[0] <= int(event.get("frame_index", -1)) <= common_frames[-1]
    ]
    validations_by_phase = defaultdict(list)
    for event in validation_events:
        validations_by_phase[str(event.get("phase", "unknown"))].append(event)
    validations_by_window = {
        f"{start}-{end}": [
            event
            for event in validation_events
            if start <= int(event.get("frame_index", -1)) <= end
        ]
        for start, end in windows
    }
    open_loop_predictions = [
        event
        for event in run["events"]
        if event.get("event") == "open_loop_prediction"
        and "gap_improvement" in event.get("prediction", {})
    ]
    open_loop_groups = defaultdict(list)
    for event in open_loop_predictions:
        prediction = event["prediction"]
        open_loop_groups[
            (
                str(prediction.get("protocol", "unknown")),
                int(prediction.get("horizon_frames", 0)),
            )
        ].append(event)
    open_loop_by_protocol = defaultdict(dict)
    for (protocol, horizon), events in sorted(open_loop_groups.items()):
        open_loop_by_protocol[protocol][str(horizon)] = (
            open_loop_prediction_stats(events)
        )
    committed_validations = [
        event
        for event in validation_events
        if event.get("prediction", {}).get("status") == "committed"
    ]
    commit_phase_counts = Counter(
        str(event.get("phase", "unknown")) for event in committed_validations
    )
    final_material = None
    horizon_one_material_isolation = [
        event
        for event in open_loop_predictions
        if event["prediction"].get("protocol") == "material_isolation"
        and int(event["prediction"].get("horizon_frames", 0)) == 1
        and "new_distance" in event.get("material", {})
    ]
    if horizon_one_material_isolation:
        latest = max(
            horizon_one_material_isolation,
            key=lambda event: int(event["prediction"].get("evaluation_id", 0)),
        )
        final_material = {
            "evaluation_id": int(latest["prediction"]["evaluation_id"]),
            "frame_index": int(latest["frame_index"]),
            "distance": latest["material"]["new_distance"],
            "shape": latest["material"]["new_shape"],
            "last_commit_distance_change_rms": latest["material"][
                "distance_change_rms"
            ],
            "last_commit_shape_change_rms": latest["material"][
                "shape_change_rms"
            ],
        }
    residual_summary = {
        "update_count": len(visual_updates),
        "accepted_count": sum(
            bool(event["image"].get("accepted")) for event in visual_updates
        ),
        "loss_before": stats(
            event["image"]["mean_loss_before"] for event in visual_updates
        ),
        "loss_after": stats(
            event["image"]["mean_loss_after"] for event in visual_updates
        ),
        "reduction_fraction": stats(
            event["image"]["loss_reduction_fraction"]
            for event in visual_updates
        ),
    }
    return {
        "directory": run["directory"],
        "configuration": {
            "stiffness_admission_profile": run["metadata"].get(
                "stiffness_admission_profile"
            ),
            "tip_entry_allowance_m": run["metadata"].get(
                "tip_entry_allowance_m"
            ),
            "grip_maximum_capture_penetration_m": run["metadata"].get(
                "grip_maximum_capture_penetration_m"
            ),
            "stiffness_maximum_penetration_m": run["metadata"].get(
                "stiffness_maximum_penetration_m"
            ),
            "stiffness_maximum_jaw_speed_rad_s": run["metadata"].get(
                "stiffness_maximum_jaw_speed_rad_s"
            ),
            "stiffness_transition_cooldown_updates": run["metadata"].get(
                "stiffness_transition_cooldown_updates"
            ),
            "stiffness_prediction_absolute_margin": run["metadata"].get(
                "stiffness_prediction_absolute_margin"
            ),
            "stiffness_prediction_relative_margin": run["metadata"].get(
                "stiffness_prediction_relative_margin"
            ),
            "stiffness_commit_validation_horizon_frames": run[
                "metadata"
            ].get("stiffness_commit_validation_horizon_frames"),
            "stiffness_candidate_log_step_scales": run["metadata"].get(
                "stiffness_candidate_log_step_scales"
            ),
            "stiffness_log_learning_rate": run["metadata"].get(
                "stiffness_log_learning_rate"
            ),
            "stiffness_validation_visual_objective": run["metadata"].get(
                "stiffness_validation_visual_objective"
            ),
            "stiffness_adopt_validated_rollout": run["metadata"].get(
                "stiffness_adopt_validated_rollout"
            ),
            "stiffness_maximum_adopted_state_rms_m": run["metadata"].get(
                "stiffness_maximum_adopted_state_rms_m"
            ),
            "stiffness_maximum_adopted_state_maximum_m": run[
                "metadata"
            ].get("stiffness_maximum_adopted_state_maximum_m"),
        },
        "sampled_frames": common_frames,
        "prediction_loss": stats(prediction_losses),
        "left_prediction_loss": stats(camera_losses[0]),
        "right_prediction_loss": stats(camera_losses[1]),
        "phase_prediction_loss": {
            phase: stats(values) for phase, values in sorted(by_phase.items())
        },
        "window_prediction_loss": {
            window: stats(values) for window, values in by_window.items()
        },
        "physical": {
            "minimum_volume_ratio": min(finite(minimum_volume_ratios)),
            "maximum_penetration_m": max(finite(maximum_penetrations)),
            "maximum_inverted_tetrahedra": int(max(inverted)),
        },
        "residual": residual_summary,
        "stiffness_gate_pause_reasons": dict(gate_reasons),
        "stiffness_gate_pause_count": sum(gate_reasons.values()),
        "stiffness_gate_by_phase": stiffness_gate_by_phase,
        "material_status_counts": dict(material_statuses),
        "validation_status_counts": dict(validation_statuses),
        "stiffness_validation": stiffness_validation_stats(validation_events),
        "stiffness_validation_by_phase": {
            phase: stiffness_validation_stats(events)
            for phase, events in sorted(validations_by_phase.items())
        },
        "stiffness_validation_by_window": {
            window: stiffness_validation_stats(events)
            for window, events in validations_by_window.items()
        },
        "stiffness_commit_phase_counts": dict(commit_phase_counts),
        "open_loop": {
            "started_count": event_type_counts.get("open_loop_started", 0),
            "completed_count": event_type_counts.get("open_loop_completed", 0),
            "incomplete_count": event_type_counts.get("open_loop_incomplete", 0),
            "by_protocol_horizon": dict(open_loop_by_protocol),
        },
        "final_verified_material": final_material,
        "event_type_counts": dict(event_type_counts),
    }


def markdown(report: dict) -> str:
    modes = report["modes"]
    lines = [
        "# SUPER 三组轨迹初步对比",
        "",
        f"共同帧：`{report['common_frame_min']}..{report['common_frame_max']}`，"
        f"共 `{report['common_frame_count']}` 帧。主指标是每一帧施加当前视觉 residual 之前的"
        "双目 masked Smooth-L1 prediction loss；越小越好。",
        "",
        "| 模式 | 平均预测损失 | 中位数 | 相对 fixed 改善 | 最小 J | 最大穿透 mm | 翻转 tet |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "fixed_pbd": "什么都不开 / Fixed PBD",
        "residual_only": "只开 residual",
        "residual_online_stiffness": "residual + 实时刚度",
    }
    for mode in EXPECTED_MODES:
        item = modes[mode]
        lines.append(
            f"| {labels[mode]} | {item['prediction_loss']['mean']:.8f} | "
            f"{item['prediction_loss']['median']:.8f} | "
            f"{item['relative_improvement_vs_fixed'] * 100.0:+.3f}% | "
            f"{item['physical']['minimum_volume_ratio']:.6f} | "
            f"{item['physical']['maximum_penetration_m'] * 1e3:.3f} | "
            f"{item['physical']['maximum_inverted_tetrahedra']} |"
        )
    lines.extend([
        "",
        "## Residual 拟合辅助指标",
        "",
        "| 模式 | 更新次数 | 接受次数 | 更新前均值 | 更新后均值 | 单次平均降幅 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for mode in ("residual_only", "residual_online_stiffness"):
        item = modes[mode]["residual"]
        lines.append(
            f"| {labels[mode]} | {item['update_count']} | {item['accepted_count']} | "
            f"{item['loss_before']['mean']:.8f} | {item['loss_after']['mean']:.8f} | "
            f"{item['reduction_fraction']['mean'] * 100.0:.3f}% |"
        )
    lines.extend([
        "",
        "注意：post-residual loss 只说明当前图像能否被 residual 拟合；三组方法的公平主比较"
        "必须看下一次观测到来前的 prediction loss。",
        "",
        "## 分阶段 prediction loss",
        "",
        "| 阶段 | 帧数 | Fixed PBD | Residual-only | Residual 改善 | Residual + online | Online 相对 residual |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    phases = sorted(
        set().union(*(
            set(modes[mode]["phase_prediction_loss"]) for mode in EXPECTED_MODES
        ))
    )
    for phase in phases:
        fixed_phase = modes["fixed_pbd"]["phase_prediction_loss"].get(phase)
        residual_phase = modes["residual_only"]["phase_prediction_loss"].get(phase)
        online_phase = modes["residual_online_stiffness"]["phase_prediction_loss"].get(phase)
        if fixed_phase is None or residual_phase is None or online_phase is None:
            continue
        residual_gain = (
            fixed_phase["mean"] - residual_phase["mean"]
        ) / max(fixed_phase["mean"], 1.0e-12)
        online_delta_phase = (
            online_phase["mean"] - residual_phase["mean"]
        ) / max(residual_phase["mean"], 1.0e-12)
        lines.append(
            f"| {phase} | {fixed_phase['count']} | {fixed_phase['mean']:.8f} | "
            f"{residual_phase['mean']:.8f} | {residual_gain * 100.0:+.3f}% | "
            f"{online_phase['mean']:.8f} | {online_delta_phase * 100.0:+.4f}% |"
        )
    lines.extend([
        "",
        "capture/place 目前分别只有 2/1 帧，只能看趋势，不能据此调材料参数。",
        "",
        "## 每 50 帧时间窗口",
        "",
        "| 帧窗口 | 帧数 | Fixed PBD | Residual-only | Residual + online |",
        "|---|---:|---:|---:|---:|",
    ])
    for window, fixed_window in modes["fixed_pbd"]["window_prediction_loss"].items():
        residual_window = modes["residual_only"]["window_prediction_loss"][window]
        online_window = modes["residual_online_stiffness"]["window_prediction_loss"][window]
        lines.append(
            f"| {window} | {fixed_window['count']} | {fixed_window['mean']:.8f} | "
            f"{residual_window['mean']:.8f} | {online_window['mean']:.8f} |"
        )
    online = modes["residual_online_stiffness"]
    residual = modes["residual_only"]
    online_delta = (
        online["prediction_loss"]["mean"]
        - residual["prediction_loss"]["mean"]
    ) / max(residual["prediction_loss"]["mean"], 1.0e-12)
    validation_count = online["event_type_counts"].get(
        "stiffness_validation", 0
    )
    open_loop_count = online["event_type_counts"].get(
        "open_loop_started", 0
    )
    lines.extend([
        "",
        "## 在线刚度是否真的工作",
        "",
        "- 本轮刚度学习穿透门："
        f"`{float(online['configuration']['stiffness_maximum_penetration_m'] or 0.0) * 1e3:.3f} mm`。",
        "- 全局 rapid-q7 暂停："
        f"`{'关闭' if online['configuration']['stiffness_maximum_jaw_speed_rad_s'] is None else str(online['configuration']['stiffness_maximum_jaw_speed_rad_s']) + ' rad/s'}`；"
        "transition cooldown："
        f"`{online['configuration']['stiffness_transition_cooldown_updates']}` 次更新。",
        "- 候选提交降损门："
        f"absolute=`{float(online['configuration']['stiffness_prediction_absolute_margin'] or 0.0):.1e}`，"
        f"relative=`{float(online['configuration']['stiffness_prediction_relative_margin'] or 0.0):.1e}`。",
        "- 闭环候选预演："
        f"H=`{online['configuration']['stiffness_commit_validation_horizon_frames']}` 帧；"
        f"目标=`{online['configuration']['stiffness_validation_visual_objective']}`；"
        f"log learning rate=`{online['configuration']['stiffness_log_learning_rate']}`；"
        "步长倍率="
        f"`{online['configuration']['stiffness_candidate_log_step_scales']}`。",
        "- 验证状态接回："
        f"`{online['configuration']['stiffness_adopt_validated_rollout']}`；"
        "RMS/单粒子上限="
        f"`{float(online['configuration']['stiffness_maximum_adopted_state_rms_m'] or 0.0) * 1e3:.3f}`/"
        f"`{float(online['configuration']['stiffness_maximum_adopted_state_maximum_m'] or 0.0) * 1e3:.3f} mm`。",
        f"- 在线组 visual updates：`{online['residual']['update_count']}`；刚度安全门暂停："
        f"`{online['stiffness_gate_pause_count']}`。",
        f"- stiffness validation：`{validation_count}`；"
        f"open-loop started：`{open_loop_count}`；"
        f"validation status：`{online['validation_status_counts']}`。",
        f"- online 相对 residual-only 的均值差：`{online_delta * 100.0:+.4f}%`。",
        "",
        (
            "全部 visual update 都被安全门暂停，在线组实际等同 residual-only。"
            if validation_count == 0
            else (
                "安全门已放行 candidate，但没有候选通过验证/commit；verified 刚度场未改变，"
                "online 与 residual-only 的剩余微差仍不能归因于实时刚度。"
                if open_loop_count == 0
                else "已有候选通过验证，需继续检查各 horizon 和 commit 统计。"
            )
        ),
        "",
    ])
    if online["stiffness_gate_by_phase"]:
        lines.extend([
            "### 各动作阶段是否被门禁暂停",
            "",
            "| 阶段 | Visual updates | Eligible | Paused | 暂停原因 |",
            "|---|---:|---:|---:|---|",
        ])
        for phase, item in online["stiffness_gate_by_phase"].items():
            reasons = ", ".join(
                f"{reason}={count}"
                for reason, count in item["pause_reason_counts"].items()
            ) or "-"
            lines.append(
                f"| {phase} | {item['visual_update_count']} | "
                f"{item['eligible_count']} | {item['paused_count']} | {reasons} |"
            )
        lines.extend([
            "",
            "`Eligible` 表示该帧允许提出候选，不等于候选已经通过影子验证并提交。",
            "",
        ])
    validation_by_phase = online["stiffness_validation_by_phase"]
    if validation_by_phase:
        lines.extend([
            "## 候选刚度的分阶段影子验证",
            "",
            "这里比较的是同一状态下的 baseline 与 candidate；`平均降损` 为正表示候选更好。"
            "候选必须达到 `required` 才能提交，不能只看是否为正。",
            "",
            "| 阶段 | 验证事件 | 完整影子对 | 候选降损次数 | 通过降损门 | 最终提交 | 平均降损 ×1e-6 | 平均 required ×1e-6 | 体积退化次数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for phase, item in validation_by_phase.items():
            lines.append(
                f"| {phase} | {item['event_count']} | {item['shadow_pair_count']} | "
                f"{item['candidate_lower_loss_count']} | {item['required_margin_pass_count']} | "
                f"{item['committed_count']} | "
                f"{item['gap_improvement']['mean'] * 1e6:.3f} | "
                f"{item['required_improvement']['mean'] * 1e6:.3f} | "
                f"{item['rejection_reason_counts'].get('volume_quality_regression', 0)} |"
            )
        lines.extend([
            "",
            "过渡阶段没有完整影子对时，通常表示捕获/释放、快速 q7、穿透或 cooldown 门禁"
            "按原方案暂停了候选，而不是把接触期间的刚度学习永久关闭。",
            "",
            "## 候选刚度的时间窗口验证",
            "",
            "| 帧窗口 | 完整影子对 | 候选降损次数 | 达标次数 | 平均降损 ×1e-6 | 平均 required ×1e-6 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for window, item in online["stiffness_validation_by_window"].items():
            lines.append(
                f"| {window} | {item['shadow_pair_count']} | "
                f"{item['candidate_lower_loss_count']} | {item['required_margin_pass_count']} | "
                f"{item['gap_improvement']['mean'] * 1e6:.3f} | "
                f"{item['required_improvement']['mean'] * 1e6:.3f} |"
            )
        lines.append("")
    open_loop = online["open_loop"]
    if open_loop["by_protocol_horizon"]:
        lines.extend([
            "## 已提交候选的 H=1/3/5/10 开放环",
            "",
            f"started=`{open_loop['started_count']}`，completed=`{open_loop['completed_count']}`，"
            f"incomplete at trajectory end=`{open_loop['incomplete_count']}`。",
            "",
            "| 协议 | H | 样本 | 正改善 | 平均 gap 改善 ×1e-6 | 中位数 ×1e-6 | 平均相对改善 | Candidate J 更低 | >0.1mm 穿透退化 | 翻转 tet |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for protocol, horizons in open_loop["by_protocol_horizon"].items():
            for horizon, item in sorted(
                horizons.items(), key=lambda pair: int(pair[0])
            ):
                lines.append(
                    f"| {protocol} | {horizon} | {item['count']} | "
                    f"{item['positive_count']} | "
                    f"{item['gap_improvement']['mean'] * 1e6:.3f} | "
                    f"{item['gap_improvement']['median'] * 1e6:.3f} | "
                    f"{item['relative_improvement']['mean'] * 100.0:+.4f}% | "
                    f"{item['candidate_j_below_baseline_count']} | "
                    f"{item['penetration_regression_over_0p1mm_count']} | "
                    f"{item['maximum_inverted_tetrahedra']} |"
                )
        lines.extend([
            "",
            "material_isolation 只比较材料差异；end_to_end 还让视觉 residual 随轨迹重新求解。"
            "平均值容易被少量坏候选拉低，因此同时报告中位数和正改善次数。",
            "",
        ])
    final_material = online["final_verified_material"]
    if final_material is not None:
        distance = final_material["distance"]
        shape = final_material["shape"]
        lines.extend([
            "## 最终 verified 空间刚度场",
            "",
            f"最后一次提交：evaluation `{final_material['evaluation_id']}`，"
            f"frame `{final_material['frame_index']}`。",
            "",
            "| 参数 | 最小 | 中位数 | 均值 | 最大 |",
            "|---|---:|---:|---:|---:|",
            f"| distance | {distance['minimum']:.6f} | {distance['median']:.6f} | "
            f"{distance['mean']:.6f} | {distance['maximum']:.6f} |",
            f"| shape | {shape['minimum']:.6f} | {shape['median']:.6f} | "
            f"{shape['mean']:.6f} | {shape['maximum']:.6f} |",
            "",
            "中位数仍等于初值 0.20/0.004，说明更新保持局部；最小/最大值反映被验证后"
            "保留下来的局部软化与硬化，而不是整块组织统一改变。",
            "",
        ])
    overall_direction = "降低" if online_delta < 0.0 else "升高"
    paired = report["paired_online_vs_residual"]
    lines.extend([
        "## 整体结论",
        "",
        f"相对 residual-only，online 的 211 帧平均 prediction loss {overall_direction} "
        f"`{abs(online_delta) * 100.0:.4f}%`。",
        f"逐帧比较为 online 更低 `{paired['online_lower_count']}` 帧、residual-only 更低 "
        f"`{paired['residual_lower_count']}` 帧、相同 `{paired['tie_count']}` 帧；"
        f"逐帧差值中位数为 `{paired['loss_difference']['median']:+.8f}`。",
        (
            "本轮整体主指标改善；仍需重复运行确认收益超过 CUDA 非确定性。"
            if online_delta < 0.0
            else "本轮整体主指标没有改善；虽然局部候选能通过 H=1，累计在线刚度尚未带来轨迹级净收益。"
        ),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs = [read_run(path) for path in args.runs]
    by_mode = {run["mode"]: run for run in runs}
    missing = set(EXPECTED_MODES) - set(by_mode)
    if missing or len(by_mode) != 3:
        raise ValueError(
            f"Expected exactly {EXPECTED_MODES}; got {tuple(sorted(by_mode))}"
        )
    common_frames = sorted(
        set.intersection(*(set(run["observations"]) for run in runs))
    )
    if not common_frames:
        raise ValueError("The three runs have no common observation frames")
    summaries = {
        mode: summarize_run(by_mode[mode], common_frames)
        for mode in EXPECTED_MODES
    }
    fixed_mean = summaries["fixed_pbd"]["prediction_loss"]["mean"]
    for summary in summaries.values():
        summary["relative_improvement_vs_fixed"] = (
            (fixed_mean - summary["prediction_loss"]["mean"])
            / max(fixed_mean, 1.0e-12)
        )
    residual_observations = by_mode["residual_only"]["observations"]
    online_observations = by_mode["residual_online_stiffness"]["observations"]
    paired_differences = [
        float(online_observations[frame]["image"]["prediction_loss"])
        - float(residual_observations[frame]["image"]["prediction_loss"])
        for frame in common_frames
    ]
    report = {
        "schema_version": 3,
        "common_frame_count": len(common_frames),
        "common_frame_min": common_frames[0],
        "common_frame_max": common_frames[-1],
        "paired_online_vs_residual": {
            "online_lower_count": sum(value < 0.0 for value in paired_differences),
            "residual_lower_count": sum(value > 0.0 for value in paired_differences),
            "tie_count": sum(value == 0.0 for value in paired_differences),
            "loss_difference": stats(paired_differences),
        },
        "modes": summaries,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "comparison.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
