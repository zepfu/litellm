"""Monotonic phase accounting. Wall time stays separate from nested spans."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from hv2.suite import PHASES
from hv2.suite.schema import empty_timing


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seconds(value: Any) -> float | None:
    """Finite durations only. Booleans and non-numeric values stay unavailable."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _duration(start: Any, end: Any) -> float | None:
    start_s = _seconds(start)
    end_s = _seconds(end)
    if start_s is None or end_s is None or end_s < start_s:
        return None
    return end_s - start_s


def _union_length(intervals: Sequence[tuple[float, float]]) -> float:
    ordered = sorted(intervals)
    if not ordered:
        return 0.0
    cursor_s, cursor_e = ordered[0]
    total = 0.0
    for start, end in ordered[1:]:
        if start <= cursor_e:
            cursor_e = max(cursor_e, end)
        else:
            total += cursor_e - cursor_s
            cursor_s, cursor_e = start, end
    total += cursor_e - cursor_s
    return total


def _sequence(row: Mapping[str, Any]) -> int | None:
    sequence = row.get("sequence")
    if sequence is None and isinstance(row.get("wait_accounting"), Mapping):
        sequence = row["wait_accounting"].get("sequence")
    if isinstance(sequence, bool):
        return None
    try:
        return int(sequence)
    except (TypeError, ValueError):
        return None


def consume_wait_checkpoints(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest wait snapshot per call id. Do not sum repeated checkpoints."""

    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        call_id = str(row.get("litellm_call_id") or row.get("call_id") or "")
        if not call_id:
            continue
        sequence_i = _sequence(row)
        if sequence_i is None:
            continue
        previous = latest.get(call_id)
        if previous is not None and sequence_i < int(previous["sequence"]):
            continue
        snapshot = row.get("wait_accounting") if isinstance(row.get("wait_accounting"), Mapping) else row
        durations = snapshot.get("durations_ms") if isinstance(snapshot, Mapping) else None
        latest[call_id] = {
            "litellm_call_id": call_id,
            "sequence": sequence_i,
            "started_at": snapshot.get("started_at") or row.get("started_at"),
            "updated_at": snapshot.get("updated_at") or row.get("updated_at"),
            "durations_ms": dict(durations) if isinstance(durations, Mapping) else {},
            "status": snapshot.get("status") or row.get("status"),
        }
    return list(latest.values())


def provider_intervals(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Attempt counts are not durations. Missing start/end stays unavailable."""

    spans: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for index, attempt in enumerate(attempts):
        ordinal = attempt.get("ordinal", index + 1)
        start = attempt.get("started_at")
        end = attempt.get("ended_at")
        duration = _duration(start, end)
        if duration is None:
            unavailable.append(f"provider_attempt:{ordinal}")
            spans.append(
                {
                    "ordinal": ordinal,
                    "duration_seconds": None,
                    "unavailable": True,
                }
            )
            continue
        spans.append(
            {
                "ordinal": ordinal,
                "started_at": start,
                "ended_at": end,
                "duration_seconds": duration,
                "unavailable": False,
            }
        )
    return {
        "attempt_count": len(attempts),
        "spans": spans,
        "unavailable": unavailable,
        "note": "attempt_count is not a duration",
    }


def _nested_spans(
    spans: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], float | None, list[str]]:
    """Sum measured nested durations. Missing spans stay listed, not inferred."""

    nested: list[dict[str, Any]] = []
    nested_sum = 0.0
    nested_known = False
    unavailable: list[str] = []
    for span in spans or []:
        copied = dict(span)
        duration = _seconds(span.get("duration_seconds"))
        if duration is not None and duration < 0:
            duration = None
        if duration is None and span.get("duration_seconds") is None:
            duration = _duration(span.get("start"), span.get("end"))
        copied["duration_seconds"] = duration
        nested.append(copied)
        if duration is None:
            unavailable.append(f"nested.{span.get('kind') or 'span'}")
            continue
        nested_sum += duration
        nested_known = True
    return nested, nested_sum if nested_known else None, unavailable


def _phase_accounts(
    phases: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float | None], list[tuple[float, float]], float, list[dict[str, Any]], list[str]]:
    """Exclusive PHASES totals plus the raw sum of measured phase intervals."""

    intervals: list[tuple[float, float]] = []
    exclusive: dict[str, float | None] = {phase: 0.0 for phase in PHASES}
    seen_phase = {phase: False for phase in PHASES}
    phase_gap = {phase: False for phase in PHASES}
    raw_sum = 0.0
    case_rows: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for row in phases:
        phase = str(row.get("phase") or "")
        start = row.get("start")
        end = row.get("end")
        duration = _duration(start, end)
        case_rows.append(
            {
                "case_id": row.get("case_id"),
                "attempt_id": row.get("attempt_id"),
                "phase": phase,
                "duration_seconds": duration,
            }
        )
        if phase not in exclusive:
            unavailable.append(f"phase.unknown:{phase}")
            continue
        seen_phase[phase] = True
        if duration is None:
            unavailable.append(f"phase.{phase}")
            phase_gap[phase] = True
            continue
        exclusive[phase] = float(exclusive[phase] or 0.0) + duration
        raw_sum += duration
        start_s = _seconds(start)
        end_s = _seconds(end)
        if start_s is not None and end_s is not None:
            intervals.append((start_s, end_s))
    for phase, seen in seen_phase.items():
        if not seen or phase_gap[phase]:
            exclusive[phase] = None
            if not seen:
                unavailable.append(f"phase.{phase}")
    return exclusive, intervals, raw_sum, case_rows, unavailable


def reconcile(
    *,
    suite_start_mono: float | None,
    suite_end_mono: float | None,
    suite_started_at: str | None,
    suite_finished_at: str | None,
    phases: Sequence[Mapping[str, Any]],
    nested_spans: Sequence[Mapping[str, Any]] | None = None,
    shared_overhead_seconds: float | None = None,
    tui_harness: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reconcile exclusive phase time against wall time without double-counting.

    ``phases`` entries use monotonic ``start``/``end`` seconds and a ``phase``
    name. Nested provider/tool/child spans are reported separately and are not
    added into ``wall_seconds``.
    """

    timing = empty_timing()
    timing["suite"]["started_at"] = suite_started_at
    timing["suite"]["finished_at"] = suite_finished_at
    wall = _duration(suite_start_mono, suite_end_mono)
    timing["suite"]["wall_seconds"] = wall
    timing["suite"]["monotonic_seconds"] = wall
    unavailable: list[str] = []
    if wall is None:
        unavailable.append("suite.wall_seconds")

    exclusive, intervals, raw_sum, case_rows, phase_unavailable = _phase_accounts(phases)
    unavailable.extend(phase_unavailable)
    union = _union_length(intervals)
    overlap = raw_sum - union
    overhead = _seconds(shared_overhead_seconds)
    if overhead is not None and overhead < 0:
        overhead = None
    timing["exclusive_phase_seconds"] = exclusive
    timing["overlap_seconds"] = overlap
    timing["shared_overhead_seconds"] = overhead
    if overhead is None:
        unavailable.append("shared_overhead_seconds")
    if wall is None or overhead is None:
        timing["unattributed_seconds"] = None
        unavailable.append("unattributed_seconds")
    else:
        timing["unattributed_seconds"] = wall - (union + overhead)
    nested, nested_sum, nested_unavailable = _nested_spans(nested_spans)
    unavailable.extend(nested_unavailable)
    timing["nested_span_sum_seconds"] = nested_sum
    timing["wall_excludes_nested_spans"] = True
    timing["cases"] = case_rows
    timing["tui_harness"] = [dict(item) for item in (tui_harness or [])]
    timing["unavailable"] = unavailable
    timing["provider_spans"] = [span for span in nested if span.get("kind") == "provider"]
    timing["tool_spans"] = [span for span in nested if span.get("kind") == "tool"]
    timing["child_spans"] = [span for span in nested if span.get("kind") == "child"]
    return timing
