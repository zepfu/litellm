"""Fixture-first local replay for malformed model/tool-call output.

Loads sanitized fixtures and invokes production detector, repair, intake, and
scorer helpers. Does not perform live provider traffic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "fixture_id",
        "source_kind",
        "provider_family",
        "transcript_surface",
        "redaction_status",
        "source_signal",
        "expected_disposition",
        "lanes",
        "tool_names",
        "sanitized_text_ref",
    }
)

ReplayLane = Literal["detect", "repair", "intake", "scorer"]
ALL_LANES: Tuple[ReplayLane, ...] = ("detect", "repair", "intake", "scorer")
EXPECTED_DISPOSITIONS = frozenset({"clean", "fail_closed", "repaired"})


class ManifestValidationError(ValueError):
    """Raised when manifest.jsonl or a manifest row fails schema checks."""


@dataclass(frozen=True)
class ManifestEntry:
    fixture_id: str
    source_kind: str
    provider_family: str
    transcript_surface: str
    redaction_status: str
    source_signal: str
    expected_disposition: str
    lanes: Tuple[str, ...]
    tool_names: Tuple[str, ...]
    sanitized_text_ref: str
    raw_source_ref: Optional[str] = None
    notes: Optional[str] = None


@dataclass(frozen=True)
class LoadedFixture:
    entry: ManifestEntry
    fixture_path: Path
    payload: Dict[str, Any]


def default_fixtures_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "model_output_replay"


def parse_manifest_line(raw: Dict[str, Any], *, line_no: int) -> ManifestEntry:
    if not isinstance(raw, dict):
        raise ManifestValidationError(f"manifest line {line_no}: expected JSON object")

    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(raw.keys()))
    if missing:
        raise ManifestValidationError(
            f"manifest line {line_no}: missing required fields: {', '.join(missing)}"
        )

    fixture_id = _require_non_empty_str(raw, "fixture_id", line_no=line_no)
    lanes = _require_str_list(raw, "lanes", line_no=line_no)
    invalid_lanes = sorted(set(lanes) - set(ALL_LANES))
    if invalid_lanes:
        raise ManifestValidationError(
            f"manifest line {line_no}: invalid lanes: {', '.join(invalid_lanes)}"
        )
    tool_names = _require_str_list(raw, "tool_names", line_no=line_no)
    sanitized_text_ref = _require_non_empty_str(raw, "sanitized_text_ref", line_no=line_no)
    expected_disposition = _require_non_empty_str(
        raw, "expected_disposition", line_no=line_no
    )
    if expected_disposition not in EXPECTED_DISPOSITIONS:
        raise ManifestValidationError(
            f"manifest line {line_no}: expected_disposition must be one of "
            f"{', '.join(sorted(EXPECTED_DISPOSITIONS))}"
        )

    raw_source_ref = raw.get("raw_source_ref")
    if raw_source_ref is not None and (
        not isinstance(raw_source_ref, str) or not raw_source_ref.strip()
    ):
        raise ManifestValidationError(
            f"manifest line {line_no}: raw_source_ref must be a non-empty string when set"
        )

    notes = raw.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ManifestValidationError(f"manifest line {line_no}: notes must be a string")

    return ManifestEntry(
        fixture_id=fixture_id,
        source_kind=_require_non_empty_str(raw, "source_kind", line_no=line_no),
        provider_family=_require_non_empty_str(raw, "provider_family", line_no=line_no),
        transcript_surface=_require_non_empty_str(
            raw, "transcript_surface", line_no=line_no
        ),
        redaction_status=_require_non_empty_str(raw, "redaction_status", line_no=line_no),
        source_signal=_require_non_empty_str(raw, "source_signal", line_no=line_no),
        expected_disposition=expected_disposition,
        lanes=tuple(lanes),
        tool_names=tuple(tool_names),
        sanitized_text_ref=sanitized_text_ref,
        raw_source_ref=raw_source_ref.strip() if isinstance(raw_source_ref, str) else None,
        notes=notes,
    )


def load_manifest(manifest_path: Path) -> List[ManifestEntry]:
    if not manifest_path.is_file():
        raise ManifestValidationError(f"manifest not found: {manifest_path}")

    entries: List[ManifestEntry] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ManifestValidationError(
                f"manifest line {line_no}: invalid JSON: {exc}"
            ) from exc
        entry = parse_manifest_line(raw, line_no=line_no)
        if entry.fixture_id in seen_ids:
            raise ManifestValidationError(
                f"manifest line {line_no}: duplicate fixture_id {entry.fixture_id!r}"
            )
        seen_ids.add(entry.fixture_id)
        entries.append(entry)
    if not entries:
        raise ManifestValidationError(f"manifest is empty: {manifest_path}")
    return entries


def load_fixture(
    entry: ManifestEntry,
    *,
    fixtures_dir: Path,
) -> LoadedFixture:
    fixture_path = (fixtures_dir / entry.sanitized_text_ref).resolve()
    base = fixtures_dir.resolve()
    if base not in fixture_path.parents and fixture_path != base:
        raise ManifestValidationError(
            f"fixture {entry.fixture_id}: sanitized_text_ref escapes fixtures dir"
        )
    if not fixture_path.is_file():
        raise ManifestValidationError(
            f"fixture {entry.fixture_id}: missing file {entry.sanitized_text_ref}"
        )
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ManifestValidationError(
            f"fixture {entry.fixture_id}: expected JSON object in {entry.sanitized_text_ref}"
        )
    if payload.get("fixture_id") != entry.fixture_id:
        raise ManifestValidationError(
            f"fixture {entry.fixture_id}: payload fixture_id mismatch"
        )
    return LoadedFixture(entry=entry, fixture_path=fixture_path, payload=payload)


def resolve_fixture(
    *,
    manifest_path: Path,
    fixture_id: str,
    fixtures_dir: Optional[Path] = None,
) -> LoadedFixture:
    base_dir = fixtures_dir or manifest_path.parent
    for entry in load_manifest(manifest_path):
        if entry.fixture_id == fixture_id:
            return load_fixture(entry, fixtures_dir=base_dir)
    raise ManifestValidationError(f"fixture_id not in manifest: {fixture_id!r}")


def _require_non_empty_str(raw: Dict[str, Any], key: str, *, line_no: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(
            f"manifest line {line_no}: {key} must be a non-empty string"
        )
    return value.strip()


def _require_str_list(raw: Dict[str, Any], key: str, *, line_no: int) -> List[str]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ManifestValidationError(
            f"manifest line {line_no}: {key} must be a non-empty list of strings"
        )
    out: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ManifestValidationError(
                f"manifest line {line_no}: {key} entries must be non-empty strings"
            )
        out.append(item.strip())
    return out


def _assistant_texts_from_response(response_body: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    output = response_body.get("output")
    if not isinstance(output, list):
        return texts
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {
                    "text",
                    "output_text",
                }:
                    text = part.get("text")
                    if isinstance(text, str):
                        texts.append(text)
    return texts


def _classify_disposition(
    *,
    malformed_detected: bool,
    repair_succeeded: bool,
) -> str:
    if not malformed_detected:
        return "clean"
    if repair_succeeded:
        return "repaired"
    return "fail_closed"


def run_replay_lane(
    loaded: LoadedFixture,
    lane: ReplayLane,
) -> Dict[str, Any]:
    """Run one replay lane using production helpers."""
    payload = loaded.payload
    response_body = payload.get("response_body")
    if not isinstance(response_body, dict):
        raise ValueError("fixture payload missing response_body object")

    request_body = payload.get("request_body")
    if request_body is not None and not isinstance(request_body, dict):
        raise ValueError("fixture request_body must be an object when present")

    adapter = str(payload.get("adapter") or "replay_harness")
    adapter_label = str(payload.get("adapter_label") or "ReplayHarness")
    adapter_model = str(payload.get("adapter_model") or response_body.get("model") or "unknown")
    intake_context = payload.get("intake_context")
    if intake_context is not None and not isinstance(intake_context, dict):
        raise ValueError("fixture intake_context must be an object when present")

    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _build_malformed_tool_call_intake_context,
        _is_codex_auto_agent_malformed_tool_call_text_output,
        _try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body,
    )
    from litellm.proxy.aawm_runtime_error_logging import (
        build_malformed_tool_call_intake_record,
        extract_malformed_tool_call_evidence,
    )
    from litellm.integrations.aawm_agent_quality_rules import score_agent_quality_context

    if lane == "detect":
        malformed = _is_codex_auto_agent_malformed_tool_call_text_output(response_body)
        evidence = extract_malformed_tool_call_evidence(response_body, max_items=3)
        return {
            "lane": lane,
            "fixture_id": loaded.entry.fixture_id,
            "malformed_detected": malformed,
            "evidence_count": len(evidence),
            "evidence_kinds": [
                item.get("detection_kind")
                for item in evidence
                if isinstance(item, dict)
            ],
        }

    if lane == "repair":
        malformed = _is_codex_auto_agent_malformed_tool_call_text_output(response_body)
        repaired = _try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body(
            response_body,
            request_body=request_body if isinstance(request_body, dict) else None,
        )
        repair_succeeded = repaired is not None and not _is_codex_auto_agent_malformed_tool_call_text_output(
            repaired
        )
        disposition = _classify_disposition(
            malformed_detected=malformed,
            repair_succeeded=repair_succeeded,
        )
        function_call_names: List[str] = []
        if isinstance(repaired, dict):
            out = repaired.get("output")
            if isinstance(out, list):
                for item in out:
                    if isinstance(item, dict) and item.get("type") == "function_call":
                        name = item.get("name")
                        if isinstance(name, str):
                            function_call_names.append(name)
        return {
            "lane": lane,
            "fixture_id": loaded.entry.fixture_id,
            "malformed_detected": malformed,
            "repair_attempted": malformed,
            "repair_succeeded": repair_succeeded,
            "disposition": disposition,
            "repaired_function_call_names": function_call_names,
            "expected_disposition": loaded.entry.expected_disposition,
        }

    if lane == "intake":
        merged_context = _build_malformed_tool_call_intake_context(
            None,
            request_body if isinstance(request_body, dict) else None,
            adapter=adapter,
            provider=(
                intake_context.get("provider")
                if isinstance(intake_context, dict)
                else None
            ),
            model_alias=(
                intake_context.get("model_alias")
                if isinstance(intake_context, dict)
                else None
            ),
        )
        if isinstance(intake_context, dict):
            merged_context.update(
                {k: v for k, v in intake_context.items() if v is not None}
            )
        evidence = extract_malformed_tool_call_evidence(response_body, max_items=5)
        record = build_malformed_tool_call_intake_record(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter=adapter,
            adapter_label=adapter_label,
            intake_context=merged_context,
            evidence_count=len(evidence),
        )
        return {
            "lane": lane,
            "fixture_id": loaded.entry.fixture_id,
            "failure_kind": record.get("failure_kind"),
            "error_code": record.get("error_code"),
            "evidence_count": len(evidence),
            "has_truncated_evidence_text": bool(
                record.get("malformed_tool_call_text")
            ),
            "route_family": record.get("route_family"),
            "model_alias": record.get("model_alias"),
        }

    if lane == "scorer":
        assistant_texts = _assistant_texts_from_response(response_body)
        result = score_agent_quality_context(assistant_texts=assistant_texts)
        fields = result.fields
        return {
            "lane": lane,
            "fixture_id": loaded.entry.fixture_id,
            "output_contract_compliance_score": fields.get(
                "output_contract_compliance_score"
            ),
            "output_contract_failure_class": fields.get("output_contract_failure_class"),
            "output_contract_failure_count": fields.get(
                "output_contract_failure_count"
            ),
            "assistant_segment_count": len(assistant_texts),
        }

    raise ValueError(f"unknown lane: {lane}")


def run_replay(
    loaded: LoadedFixture,
    lanes: Sequence[ReplayLane],
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "fixture_id": loaded.entry.fixture_id,
        "source_kind": loaded.entry.source_kind,
        "provider_family": loaded.entry.provider_family,
        "transcript_surface": loaded.entry.transcript_surface,
        "source_signal": loaded.entry.source_signal,
        "redaction_status": loaded.entry.redaction_status,
        "expected_disposition": loaded.entry.expected_disposition,
        "tool_names": list(loaded.entry.tool_names),
        "lanes": {},
    }
    for lane in lanes:
        if lane not in ALL_LANES:
            raise ValueError(f"invalid lane: {lane}")
        results["lanes"][lane] = run_replay_lane(loaded, lane)

    repair_lane = results["lanes"].get("repair")
    if isinstance(repair_lane, dict):
        results["disposition"] = repair_lane.get("disposition")
    elif isinstance(results["lanes"].get("detect"), dict):
        detect = results["lanes"]["detect"]
        if detect.get("malformed_detected"):
            results["disposition"] = "fail_closed"
        else:
            results["disposition"] = "clean"
    return results


def compact_public_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Strip fields that may embed raw transcript snippets."""
    forbidden_keys = {
        "malformed_tool_call_text",
        "malformed_tool_call_payload",
        "malformed_tool_call_evidence",
        "preview",
    }

    def _scrub(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                key: _scrub(value)
                for key, value in obj.items()
                if key not in forbidden_keys
            }
        if isinstance(obj, list):
            return [_scrub(item) for item in obj]
        if isinstance(obj, str):
            lowered = obj.lower()
            if "tool label:" in lowered or "input payload:" in lowered:
                return "[redacted_transcript_string]"
        return obj

    return _scrub(result)
