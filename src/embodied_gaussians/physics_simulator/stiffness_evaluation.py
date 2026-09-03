"""Persistent metrics and action phases for online stiffness evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping


STIFFNESS_ACTION_PHASES = (
    "press",
    "close",
    "capture",
    "lift",
    "place",
    "release",
    "idle",
)


def parse_stiffness_evaluation_horizons(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, positive and unique video-frame horizon list."""
    try:
        horizons = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(
            "Stiffness evaluation horizons must be comma-separated integers"
        ) from error
    if not horizons:
        raise ValueError("At least one stiffness evaluation horizon is required")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("Stiffness evaluation horizons must be positive")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Stiffness evaluation horizons must be unique")
    return tuple(sorted(horizons))


@dataclass
class StiffnessActionPhaseClassifier:
    """Classify runtime samples into the six phases required by the plan.

    World +Z is the table-normal/up direction in the SUPER scene.  Capture and
    release transitions are held for a few samples so a short transition is
    still attached to the next visual/metric record.
    """

    vertical_deadband_m: float = 2.0e-5
    transition_hold_samples: int = 3

    def __post_init__(self) -> None:
        if self.vertical_deadband_m < 0.0:
            raise ValueError("Phase vertical deadband cannot be negative")
        if self.transition_hold_samples < 1:
            raise ValueError("Phase transition hold must be positive")
        self.reset()

    def reset(self) -> None:
        self.previous_tool_height_m: float | None = None
        self.previous_grip_active = False
        self.transport_phase = "lift"
        self.held_transition: str | None = None
        self.held_transition_remaining = 0
        self.last_phase = "idle"

    def classify(
        self,
        contact_metrics: Mapping[str, Any] | None,
        tool_height_m: float | None,
    ) -> str:
        metrics = contact_metrics or {}
        grip_active = bool(metrics.get("persistent_grip_active", False))
        motion = str(
            metrics.get("persistent_grip_q7_motion_state", "unknown")
        ).lower()
        activation_counter = int(
            metrics.get("persistent_grip_activation_counter", 0)
        )
        contact_count = int(metrics.get("contact_count", 0))
        top_contact_count = int(metrics.get("top_barrier_contact_count", 0))

        delta_height = 0.0
        if (
            tool_height_m is not None
            and self.previous_tool_height_m is not None
            and math.isfinite(tool_height_m)
            and math.isfinite(self.previous_tool_height_m)
        ):
            delta_height = tool_height_m - self.previous_tool_height_m

        captured = grip_active and not self.previous_grip_active
        released = self.previous_grip_active and not grip_active
        if captured:
            phase = "capture"
            self.held_transition = phase
            self.held_transition_remaining = self.transition_hold_samples
            self.transport_phase = "lift"
        elif released:
            phase = "release"
            self.held_transition = phase
            self.held_transition_remaining = self.transition_hold_samples
        elif self.held_transition_remaining > 0 and self.held_transition is not None:
            phase = self.held_transition
            self.held_transition_remaining -= 1
        elif not grip_active and (motion == "closing" or activation_counter > 0):
            phase = "close"
        elif grip_active:
            if motion == "opening":
                phase = "release"
            elif delta_height > self.vertical_deadband_m:
                phase = "lift"
                self.transport_phase = phase
            elif delta_height < -self.vertical_deadband_m:
                phase = "place"
                self.transport_phase = phase
            else:
                phase = self.transport_phase
        elif motion == "opening":
            phase = "release"
        elif (
            delta_height < -self.vertical_deadband_m
            or contact_count > 0
            or top_contact_count > 0
        ):
            phase = "press"
        else:
            phase = "idle"

        if tool_height_m is not None and math.isfinite(tool_height_m):
            self.previous_tool_height_m = float(tool_height_m)
        self.previous_grip_active = grip_active
        self.last_phase = phase
        return phase


@dataclass
class _RunningStatistic:
    count: int = 0
    total: float = 0.0
    minimum: float = float("inf")
    maximum: float = -float("inf")

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.total / self.count if self.count else 0.0,
            "minimum": self.minimum if self.count else 0.0,
            "maximum": self.maximum if self.count else 0.0,
        }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    return str(value)


def _flatten_numeric(
    value: Any,
    *,
    prefix: str = "",
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(item, prefix=child))
        return flattened
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            flattened.update(_flatten_numeric(item, prefix=child))
        return flattened
    if isinstance(value, bool) or value is None:
        return flattened
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            flattened[prefix] = numeric
        return flattened
    item = getattr(value, "item", None)
    if callable(item):
        try:
            numeric = float(item())
        except (TypeError, ValueError, RuntimeError):
            return flattened
        if math.isfinite(numeric):
            flattened[prefix] = numeric
    return flattened


def _flatten_categorical(
    value: Any,
    *,
    prefix: str = "",
) -> dict[str, str]:
    flattened: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_categorical(item, prefix=child))
        return flattened
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            flattened.update(_flatten_categorical(item, prefix=child))
        return flattened
    if isinstance(value, bool):
        flattened[prefix] = "true" if value else "false"
    elif isinstance(value, str) and value:
        flattened[prefix] = value
    return flattened


class StiffnessMetricsRecorder:
    """Write append-only events and continuously recoverable phase summaries."""

    schema_version = 1

    def __init__(
        self,
        output_directory: Path,
        *,
        metadata: Mapping[str, Any],
        summary_interval_events: int = 10,
    ) -> None:
        self.output_directory = Path(output_directory).resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_directory / "events.jsonl"
        self.summary_path = self.output_directory / "summary.json"
        self.metadata_path = self.output_directory / "metadata.json"
        existing = [
            path
            for path in (self.events_path, self.summary_path, self.metadata_path)
            if path.exists()
        ]
        if existing:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"Stiffness evaluation output is not empty: {names}"
            )
        if summary_interval_events < 1:
            raise ValueError("Summary interval must be positive")
        self.summary_interval_events = int(summary_interval_events)
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.metadata = {
            "schema_version": self.schema_version,
            "started_utc": self.started_utc,
            **_jsonable(dict(metadata)),
        }
        self.metadata_path.write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._events_file = self.events_path.open(
            "x", encoding="utf-8", buffering=1
        )
        self.event_count = 0
        self.phase_counts = {phase: 0 for phase in STIFFNESS_ACTION_PHASES}
        self.event_type_counts: dict[str, int] = {}
        self._statistics: dict[
            str, dict[str, dict[str, _RunningStatistic]]
        ] = {}
        self._categorical_counts: dict[
            str, dict[str, dict[str, dict[str, int]]]
        ] = {}
        self._closed = False
        self.write_summary()

    def record(
        self,
        *,
        event: str,
        frame_index: int,
        timestamp_s: float,
        phase: str,
        image: Mapping[str, Any] | None = None,
        physical: Mapping[str, Any] | None = None,
        material: Mapping[str, Any] | None = None,
        prediction: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        force_summary: bool = False,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Cannot record after stiffness metrics close")
        normalized_phase = phase if phase in STIFFNESS_ACTION_PHASES else "idle"
        record = {
            "schema_version": self.schema_version,
            "event_index": self.event_count,
            "event": str(event),
            "frame_index": int(frame_index),
            "timestamp_s": float(timestamp_s),
            "phase": normalized_phase,
            "image": _jsonable(image or {}),
            "physical": _jsonable(physical or {}),
            "material": _jsonable(material or {}),
            "prediction": _jsonable(prediction or {}),
            "details": _jsonable(details or {}),
        }
        self._events_file.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._events_file.flush()
        self.event_count += 1
        self.phase_counts[normalized_phase] += 1
        self.event_type_counts[event] = self.event_type_counts.get(event, 0) + 1

        phase_stats = self._statistics.setdefault(normalized_phase, {})
        phase_categories = self._categorical_counts.setdefault(
            normalized_phase, {}
        )
        for category in ("image", "physical", "material", "prediction"):
            category_stats = phase_stats.setdefault(category, {})
            for key, value in _flatten_numeric(record[category]).items():
                category_stats.setdefault(key, _RunningStatistic()).add(value)
            category_counts = phase_categories.setdefault(category, {})
            for key, value in _flatten_categorical(record[category]).items():
                values = category_counts.setdefault(key, {})
                values[value] = values.get(value, 0) + 1

        if force_summary or self.event_count % self.summary_interval_events == 0:
            self.write_summary()
        return record

    def summary(self) -> dict[str, Any]:
        phase_metrics: dict[str, Any] = {}
        for phase, categories in self._statistics.items():
            phase_metrics[phase] = {
                category: {
                    key: statistic.as_dict()
                    for key, statistic in sorted(statistics.items())
                }
                for category, statistics in categories.items()
            }
        return {
            "schema_version": self.schema_version,
            "started_utc": self.started_utc,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "event_count": self.event_count,
            "phase_counts": self.phase_counts,
            "event_type_counts": self.event_type_counts,
            "phase_metrics": phase_metrics,
            "phase_categorical_counts": self._categorical_counts,
        }

    def write_summary(self) -> None:
        temporary = self.output_directory / "summary.json.tmp"
        temporary.write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.summary_path)

    def close(self) -> None:
        if self._closed:
            return
        self.write_summary()
        self._events_file.close()
        self._closed = True

    def __enter__(self) -> "StiffnessMetricsRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
