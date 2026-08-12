"""Strict, data-minimizing parser for AAWM Codex review decisions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping, Optional

AAWM_REVIEW_MAX_OUTPUT_ITEMS = 32
AAWM_REVIEW_MAX_CONTENT_ITEMS = 32
AAWM_REVIEW_MAX_OUTPUT_TEXT_CHARS = 4096
AAWM_REVIEW_MAX_RATIONALE_CHARS = 512
AAWM_REVIEW_MAX_CLASSIFICATION_CHARS = 16

_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{6,}")
_TOKEN_PREFIX_RE = re.compile(
    r"(?i)\b(?:sk|pk|xai)-[A-Za-z0-9._~+/=-]{6,}|\bya29\.[A-Za-z0-9._~+/=-]{6,}"
)
_NAMED_SECRET_RE = re.compile(
    r"(?i)\b(authorization|api[ _-]?key|access[ _-]?token|password|secret)"
    r"\s*[:=]\s*([^\s,;]+)"
)


class AawmReviewDecisionParseFailure(str, Enum):
    RESPONSE_NOT_OBJECT = "response_not_object"
    RESPONSE_NOT_COMPLETED = "response_not_completed"
    OUTPUT_NOT_LIST = "output_not_list"
    OUTPUT_TOO_LARGE = "output_too_large"
    CONTENT_TOO_LARGE = "content_too_large"
    NON_ASSISTANT_OUTPUT_TEXT = "non_assistant_output_text"
    OUTPUT_TEXT_MISSING = "output_text_missing"
    OUTPUT_TEXT_MULTIPLE = "output_text_multiple"
    OUTPUT_TEXT_INVALID = "output_text_invalid"
    OUTPUT_TEXT_TOO_LARGE = "output_text_too_large"
    JSON_MALFORMED = "json_malformed"
    JSON_TRAILING_CONTENT = "json_trailing_content"
    DECISION_NOT_OBJECT = "decision_not_object"
    DUPLICATE_OUTCOME = "duplicate_outcome"
    DUPLICATE_FIELD = "duplicate_field"
    UNEXPECTED_FIELD = "unexpected_field"
    OUTCOME_INVALID = "outcome_invalid"
    RISK_LEVEL_INVALID = "risk_level_invalid"
    USER_AUTHORIZATION_INVALID = "user_authorization_invalid"
    RATIONALE_INVALID = "rationale_invalid"
    RATIONALE_TOO_LARGE = "rationale_too_large"


@dataclass(frozen=True)
class AawmReviewDecision:
    outcome: Literal["allow", "deny"]
    rationale: Optional[str] = None


@dataclass(frozen=True)
class AawmReviewDecisionParseResult:
    decision: Optional[AawmReviewDecision] = None
    failure: Optional[AawmReviewDecisionParseFailure] = None

    @property
    def ok(self) -> bool:
        return self.decision is not None and self.failure is None


class _DuplicateJsonField(ValueError):
    def __init__(self, field_name: str) -> None:
        super().__init__(field_name)
        self.field_name = field_name


def _reject(
    failure: AawmReviewDecisionParseFailure,
) -> AawmReviewDecisionParseResult:
    return AawmReviewDecisionParseResult(failure=failure)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonField(key)
        value[key] = item
    return value


def _sanitize_rationale(value: str) -> Optional[str]:
    without_ansi = _ANSI_ESCAPE_RE.sub("", value)
    printable = "".join(
        char if char.isprintable() and char not in "\r\n\t" else " "
        for char in without_ansi
    )
    cleaned = " ".join(printable.split())
    if not cleaned:
        return None
    cleaned = _BEARER_RE.sub("Bearer [REDACTED]", cleaned)
    cleaned = _TOKEN_PREFIX_RE.sub("[REDACTED]", cleaned)
    cleaned = _NAMED_SECRET_RE.sub(
        lambda match: f"{match.group(1)}=[REDACTED]",
        cleaned,
    )
    return cleaned or None


def _valid_bounded_enum(
    value: Any,
    allowed_values: set[str],
) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= AAWM_REVIEW_MAX_CLASSIFICATION_CHARS
        and value in allowed_values
    )


def _extract_single_assistant_output_text(
    response: Mapping[str, Any],
) -> tuple[Optional[str], Optional[AawmReviewDecisionParseFailure]]:
    output = response.get("output")
    if not isinstance(output, list):
        return None, AawmReviewDecisionParseFailure.OUTPUT_NOT_LIST
    if len(output) > AAWM_REVIEW_MAX_OUTPUT_ITEMS:
        return None, AawmReviewDecisionParseFailure.OUTPUT_TOO_LARGE

    output_texts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(content, list):
            continue
        if len(content) > AAWM_REVIEW_MAX_CONTENT_ITEMS:
            return None, AawmReviewDecisionParseFailure.CONTENT_TOO_LARGE
        for content_item in content:
            if (
                not isinstance(content_item, Mapping)
                or content_item.get("type") != "output_text"
            ):
                continue
            if role != "assistant":
                return (
                    None,
                    AawmReviewDecisionParseFailure.NON_ASSISTANT_OUTPUT_TEXT,
                )
            text = content_item.get("text")
            if not isinstance(text, str) or not text.strip():
                return None, AawmReviewDecisionParseFailure.OUTPUT_TEXT_INVALID
            if len(text) > AAWM_REVIEW_MAX_OUTPUT_TEXT_CHARS:
                return None, AawmReviewDecisionParseFailure.OUTPUT_TEXT_TOO_LARGE
            output_texts.append(text)
            if len(output_texts) > 1:
                return None, AawmReviewDecisionParseFailure.OUTPUT_TEXT_MULTIPLE

    if not output_texts:
        return None, AawmReviewDecisionParseFailure.OUTPUT_TEXT_MISSING
    return output_texts[0], None


def parse_aawm_review_decision(
    response: Any,
) -> AawmReviewDecisionParseResult:
    """Return only a sanitized decision or a bounded failure classification."""

    if not isinstance(response, Mapping):
        return _reject(AawmReviewDecisionParseFailure.RESPONSE_NOT_OBJECT)
    if response.get("status") != "completed":
        return _reject(AawmReviewDecisionParseFailure.RESPONSE_NOT_COMPLETED)

    output_text, extraction_failure = _extract_single_assistant_output_text(response)
    if extraction_failure is not None:
        return _reject(extraction_failure)
    assert output_text is not None

    decoder = json.JSONDecoder(object_pairs_hook=_unique_json_object)
    stripped = output_text.lstrip()
    try:
        decision_value, end_index = decoder.raw_decode(stripped)
    except _DuplicateJsonField as exc:
        if exc.field_name == "outcome":
            return _reject(AawmReviewDecisionParseFailure.DUPLICATE_OUTCOME)
        return _reject(AawmReviewDecisionParseFailure.DUPLICATE_FIELD)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _reject(AawmReviewDecisionParseFailure.JSON_MALFORMED)

    if stripped[end_index:].strip():
        return _reject(AawmReviewDecisionParseFailure.JSON_TRAILING_CONTENT)
    if not isinstance(decision_value, dict):
        return _reject(AawmReviewDecisionParseFailure.DECISION_NOT_OBJECT)

    allowed_fields = {
        "outcome",
        "rationale",
        "risk_level",
        "user_authorization",
    }
    if any(field not in allowed_fields for field in decision_value):
        return _reject(AawmReviewDecisionParseFailure.UNEXPECTED_FIELD)

    outcome = decision_value.get("outcome")
    if outcome not in {"allow", "deny"}:
        return _reject(AawmReviewDecisionParseFailure.OUTCOME_INVALID)

    if "risk_level" in decision_value and not _valid_bounded_enum(
        decision_value["risk_level"],
        {"low", "medium", "high", "critical"},
    ):
        return _reject(AawmReviewDecisionParseFailure.RISK_LEVEL_INVALID)

    if "user_authorization" in decision_value and not _valid_bounded_enum(
        decision_value["user_authorization"],
        {"unknown", "low", "medium", "high"},
    ):
        return _reject(AawmReviewDecisionParseFailure.USER_AUTHORIZATION_INVALID)

    raw_rationale = decision_value.get("rationale")
    if raw_rationale is None:
        rationale = None
    elif not isinstance(raw_rationale, str):
        return _reject(AawmReviewDecisionParseFailure.RATIONALE_INVALID)
    else:
        if len(raw_rationale) > AAWM_REVIEW_MAX_RATIONALE_CHARS:
            return _reject(AawmReviewDecisionParseFailure.RATIONALE_TOO_LARGE)
        rationale = _sanitize_rationale(raw_rationale)
        if rationale is not None and len(rationale) > AAWM_REVIEW_MAX_RATIONALE_CHARS:
            return _reject(AawmReviewDecisionParseFailure.RATIONALE_TOO_LARGE)

    return AawmReviewDecisionParseResult(
        decision=AawmReviewDecision(
            outcome=outcome,
            rationale=rationale,
        )
    )


__all__ = [
    "AAWM_REVIEW_MAX_CLASSIFICATION_CHARS",
    "AAWM_REVIEW_MAX_CONTENT_ITEMS",
    "AAWM_REVIEW_MAX_OUTPUT_ITEMS",
    "AAWM_REVIEW_MAX_OUTPUT_TEXT_CHARS",
    "AAWM_REVIEW_MAX_RATIONALE_CHARS",
    "AawmReviewDecision",
    "AawmReviewDecisionParseFailure",
    "AawmReviewDecisionParseResult",
    "parse_aawm_review_decision",
]
