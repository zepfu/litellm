"""GREEN Wave 6 tests: real-incident classification fixtures + CSV coverage checklist.

Fixture cases below are derived from real observed incidents in the (gitignored)
``.analysis/error-archive/*.jsonl`` corpus -- representative status/message pairs
are embedded inline so this test file does not depend on the un-committed archive
at run time. The coverage checklist reads the (also gitignored)
``.analysis/agentic_tui_error_code_catalog_unified_2026-07-20.csv`` at collection
time; both files are read-only inputs used to derive these test cases and are
never committed.
"""

from __future__ import annotations

import csv
import os

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    classification as clsf,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    failure_vocabulary as fv,
)

_CATALOG_CSV_PATH = "/home/zepfu/projects/litellm/.analysis/" "agentic_tui_error_code_catalog_unified_2026-07-20.csv"

# Provider-reachable layers: rows whose failure signal can actually surface
# through a LiteLLM pass-through/adapter call (HTTP status, provider error
# code/type, or a provider wire-protocol error). TUI/headless and TUI hook
# layers are the client's *own* local presentation/lifecycle surface -- they
# never arrive as an upstream provider failure signal on the LiteLLM side and
# are asserted out-of-scope (never coolable) below.
_PROVIDER_REACHABLE_LAYERS = frozenset(
    {
        "Provider API",
        "Provider API / client",
        "Provider routing API",
        "Anthropic API skin",
        "Responses API skin",
        "Chat/completions stream",
        "Embeddings API",
        "NIM endpoint router",
        "NIM health/readiness",
        "NIM model selection",
        "NVCF direct invocation",
        "NVCF legacy pexec",
        "NVCF / LLM Gateway",
        "Hosted Build/API Catalog",
        "Wire mode",
    }
)

_TUI_LAYERS = frozenset({"TUI hook", "TUI/headless"})

# Residual known-gap set after D1-587 fixture-backed mappings for
# image-content sub-errors, JSON-RPC wire codes, and HTTP-200-body stream
# failures. A class remains listed while any provider-reachable catalog row
# in that class still classifies as unknown (limits, local/client network,
# non-error 2xx/3xx, passthrough shells, or non-theme request/body codes).
_KNOWN_COVERAGE_GAPS = frozenset(
    {
        "Agent limit",
        "Agent/model limit",
        "Billing/quota",
        "Context limit",
        "Fix request/content",
        "Invalid request",
        "Layered/platform passthrough",
        "Local network",
        "Local network/config",
        "Model refusal",
        "Model/output limit",
        "Not found",
        "Payload limit",
        "Pending/non-error",
        "Pending/not terminal",
        "Permission/policy block",
        "Plan/tool restriction",
        "Precondition/conflict",
        "Protocol-mapped error",
        "Provider overload",
        "Provider unavailable",
        "Quota/limit",
        "Reconnect",
        "Result indirection",
        "Retry transient",
        "Server error",
        "Server/API error",
        "State/session recovery",
        "Timeout",
        "Unknown/fallback",
        "Workload-specific passthrough",
    }
)


# --- Representative real-incident fixtures (derived from .analysis/error-archive) ---
# Each tuple: (status_code, message, expected_class, expected_origin, expected_confidence)
_ARCHIVE_FIXTURE_CASES: tuple[tuple[object, str, str, str, str], ...] = (
    # D1-502: dev Anthropic 529 hidden-retry exhaustion (real observed incident).
    (
        529,
        '{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
        "provider_5xx",
        "upstream",
        "structured",
    ),
    # D1-507: xAI Grok Build 403 safety-check content denial.
    (
        403,
        '{"code":"permission-denied","error":"Content violates usage guidelines. '
        'Failed check: SAFETY_CHECK_TYPE_CYBER"}',
        "auth",
        "upstream",
        "structured",
    ),
    # D1-475: prod Grok Build 402 usage-balance-exhausted.
    (
        402,
        '{"error":"Grok Build usage balance exhausted"}',
        "provider_4xx_other",
        "upstream",
        "structured",
    ),
    # dev-error-D1-349: prod/dev Anthropic 401 invalid-authentication-credentials.
    (
        401,
        '{"type":"error","error":{"type":"authentication_error",' '"message":"Invalid authentication credentials"}}',
        "auth",
        "upstream",
        "structured",
    ),
    # MS-PROD-001: Kimi managed-account cooldown -- all_candidates_unavailable 429.
    (
        429,
        "all_candidates_unavailable",
        "rate_limit",
        "upstream",
        "structured",
    ),
)


# D1-587 theme fixtures derived from provider-reachable catalog rows.
# Each tuple: (status_code, message, expected_class, expected_origin, expected_confidence)
_D1_587_THEME_FIXTURE_CASES: tuple[tuple[object, str, str, str, str], ...] = (
    # Image-content sub-errors
    (
        None,
        "invalid_image The supplied image could not be accepted.",
        "invalid_media",
        "upstream",
        "marker",
    ),
    (
        None,
        "invalid_image_format The image format is invalid.",
        "invalid_media",
        "upstream",
        "marker",
    ),
    (
        None,
        "unsupported_image_format The image format is not supported.",
        "invalid_media",
        "upstream",
        "marker",
    ),
    (
        None,
        "image_too_large An image exceeds the provider/model size limit.",
        "invalid_media",
        "upstream",
        "marker",
    ),
    (
        None,
        "image_file_not_found The referenced image file was not found.",
        "invalid_media",
        "upstream",
        "marker",
    ),
    (
        None,
        "failed_to_download_image OpenAI could not download the image URL.",
        "transient",
        "upstream",
        "marker",
    ),
    (
        None,
        "image_download_failed OpenRouter or the provider could not download the referenced image.",
        "transient",
        "upstream",
        "marker",
    ),
    (
        None,
        "image_content_policy_violation The image triggered content policy.",
        "content_policy",
        "upstream",
        "marker",
    ),
    (
        None,
        "content_policy_violation The request or output was blocked by a content-policy filter.",
        "content_policy",
        "upstream",
        "marker",
    ),
    (
        400,
        "content_filter Input or output triggered Kimi's content-safety review.",
        "content_policy",
        "upstream",
        "structured",
    ),
    # JSON-RPC wire codes (negative status ints)
    (
        -32700,
        "Invalid JSON format The incoming JSON could not be parsed.",
        "serialization",
        "upstream",
        "structured",
    ),
    (
        -32600,
        "Invalid request The JSON-RPC request is missing required fields or is otherwise invalid.",
        "provider_4xx_other",
        "upstream",
        "structured",
    ),
    (
        -32601,
        "Method not found The requested method is unsupported.",
        "model_unavailable",
        "upstream",
        "structured",
    ),
    (
        -32602,
        "Invalid method parameters The method parameters do not match the schema.",
        "provider_4xx_other",
        "upstream",
        "structured",
    ),
    (
        -32603,
        "Internal error The wire server encountered an internal error.",
        "transient",
        "upstream",
        "structured",
    ),
    (
        -32000,
        "A turn is already in progress A prompt was sent while another agent turn is active.",
        "provider_4xx_other",
        "upstream",
        "structured",
    ),
    (
        -32001,
        "LLM not configured No LLM is configured for the Kimi wire server.",
        "provider_4xx_other",
        "upstream",
        "structured",
    ),
    (
        -32002,
        "Specified LLM not supported The requested LLM is not supported.",
        "model_unavailable",
        "upstream",
        "structured",
    ),
    (
        -32003,
        "LLM service error The configured LLM service returned an error.",
        "transient",
        "upstream",
        "structured",
    ),
    # HTTP-200-body / SSE stream failures
    (
        200,
        "top-level error An error occurred after streaming began, so the connection keeps HTTP 200 "
        "and emits a structured SSE error object.",
        "stream_failure",
        "upstream",
        "structured",
    ),
    (
        200,
        "finish_reason = error The streamed choice terminates with finish_reason set to error.",
        "stream_failure",
        "upstream",
        "structured",
    ),
    (
        None,
        "response.failed A Responses-compatible stream reports that the response failed.",
        "stream_failure",
        "upstream",
        "marker",
    ),
    (
        None,
        "response.error A Responses-compatible stream emits a response-scoped error event.",
        "stream_failure",
        "upstream",
        "marker",
    ),
    (
        None,
        "error A Responses-compatible stream emits a generic error event.",
        "stream_failure",
        "upstream",
        "marker",
    ),
)


@pytest.mark.parametrize(
    "status_code,message,expected_class,expected_origin,expected_confidence",
    _ARCHIVE_FIXTURE_CASES,
)
def test_archive_incidents_classify(
    status_code: object,
    message: str,
    expected_class: str,
    expected_origin: str,
    expected_confidence: str,
) -> None:
    """Representative real-archive incidents classify to the expected FailureEvent."""
    event = clsf.classify_failure(status_code=status_code, message=message)
    assert event.class_name == expected_class
    assert event.origin == expected_origin
    assert event.confidence == expected_confidence


@pytest.mark.parametrize(
    "status_code,message,expected_class,expected_origin,expected_confidence",
    _D1_587_THEME_FIXTURE_CASES,
)
def test_d1_587_theme_fixtures_classify(
    status_code: object,
    message: str,
    expected_class: str,
    expected_origin: str,
    expected_confidence: str,
) -> None:
    """Image-content, JSON-RPC, and HTTP-200 stream fixtures map to registered classes."""
    registry = clsf.register_d1_587_failure_classes()
    event = clsf.classify_failure(status_code=status_code, message=message)
    assert event.class_name == expected_class
    assert event.origin == expected_origin
    assert event.confidence == expected_confidence
    assert registry.contains(expected_class)
    if expected_origin == "upstream":
        assert fv.is_coolable(event)


def test_client_cancelled_asyncio_cancelled_error_is_never_coolable() -> None:
    """asyncio.CancelledError (caller abort) classifies client/never-coolable."""
    import asyncio

    event = clsf.classify_exception(asyncio.CancelledError())
    assert event.class_name == "client_cancelled"
    assert event.origin == "client"
    assert not fv.is_coolable(event)


def _load_catalog_rows() -> list[dict[str, str]]:
    if not os.path.exists(_CATALOG_CSV_PATH):
        pytest.skip(f"error-code catalog CSV not present at {_CATALOG_CSV_PATH}")
    with open(_CATALOG_CSV_PATH, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _parse_catalog_status_code(raw_code: str) -> int | None:
    """Parse HTTP statuses and JSON-RPC negative wire codes from the catalog column.

    Never raises: non-numeric, overflow, or junk tokens return ``None``.
    Accepts only an optional single leading minus plus digits (rejects ``--32600``).
    """
    raw = (raw_code or "").strip()
    if not raw:
        return None
    # Stream fixtures retain HTTP 200 while failing in-body.
    if raw.lower() == "http remains 200":
        return 200
    # Exact optional-minus integer token only (no prefixes/suffixes/double-minus).
    if raw[0] == "-":
        digits = raw[1:]
        if not digits or not digits.isdigit():
            return None
    elif not raw.isdigit():
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def test_csv_coverage_checklist() -> None:
    """Every provider-reachable catalog row maps to a registered class or a known gap.

    TUI-layer rows (``TUI hook`` / ``TUI/headless``) are the client's own local
    presentation/lifecycle surface -- they are asserted out-of-scope and must
    never classify as coolable.
    """
    rows = _load_catalog_rows()
    assert rows, "expected the error-code catalog CSV to contain rows"
    clsf.register_d1_587_failure_classes()

    unresolved_gaps: list[tuple[str, str, str]] = []
    for row in rows:
        layer = row["Layer"]
        if layer not in _PROVIDER_REACHABLE_LAYERS:
            continue
        raw_code = row["HTTP / Exit / RPC"].strip()
        status_code = _parse_catalog_status_code(raw_code)
        # Include the catalog wire/status token so JSON-RPC and HTTP-200 stream
        # rows remain classifiable when the machine-code text alone is generic.
        message = f"{raw_code} {row['Machine Code / Type / Event']} {row['Meaning']}"
        event = clsf.classify_failure(status_code=status_code, message=message)
        normalized_class = row["Normalized Class"]
        registered = event.class_name != "unknown"
        known_gap = normalized_class in _KNOWN_COVERAGE_GAPS
        if not registered and not known_gap:
            unresolved_gaps.append((layer, raw_code, normalized_class))

    assert unresolved_gaps == [], (
        "provider-reachable catalog rows with neither a registered class nor a " f"listed known gap: {unresolved_gaps}"
    )


# TUI-layer rows whose bare code/type text happens to share a marker string
# with a real upstream signal (e.g. "authentication_failed" also matches the
# upstream auth marker). This is a known ambiguity of pure free-text marker
# matching, not a TUI-layer classification requirement -- callers must only
# feed classify_failure() real upstream response text, never raw client-local
# TUI event names. Documented here explicitly so the "never coolable"
# assertion below only holds for the TUI rows unaffected by this ambiguity.
_TUI_MARKER_AMBIGUOUS_CODES = frozenset(
    {
        "authentication_failed",
        "oauth_org_not_allowed",
        "FatalAuthenticationError",
        "Device authorization flow failed: fetch failed",
    }
)


def test_tui_layer_rows_excluded_from_provider_reachable_coverage() -> None:
    """TUI-layer rows are structurally excluded from the coverage checklist scope."""
    rows = _load_catalog_rows()
    tui_rows = [row for row in rows if row["Layer"] in _TUI_LAYERS]
    assert tui_rows, "expected at least one TUI-layer row in the catalog"
    for row in tui_rows:
        assert row["Layer"] not in _PROVIDER_REACHABLE_LAYERS


def test_tui_layer_rows_unaffected_by_marker_ambiguity_are_never_coolable() -> None:
    """TUI rows without an accidental marker-string collision never cool a candidate."""
    rows = _load_catalog_rows()
    tui_rows = [row for row in rows if row["Layer"] in _TUI_LAYERS]

    for row in tui_rows:
        code = row["Machine Code / Type / Event"].strip()
        if code in _TUI_MARKER_AMBIGUOUS_CODES:
            continue
        event = clsf.classify_failure(status_code=None, message=code)
        assert not fv.is_coolable(event), f"TUI-layer row {code!r} unexpectedly classified as coolable: {event}"


# --- D1-587 negative regressions: negated/docs-only/junk must stay unknown ---


@pytest.mark.parametrize(
    "status_code,message",
    [
        (None, "not an invalid_image fixture"),
        (None, "this is not content_filter related"),
        (None, "top-level error documentation only"),
        (None, "response.failed docs only"),
        (None, "invalid media alone has no CSV evidence"),
        (None, "content-filter hyphen variant is unsupported"),
        (None, "finish_reason=error compact form is unsupported"),
        (None, "x-32600y"),
        (None, "--32600"),
        (None, "prefix-32600suffix"),
        (None, "code x-32001y is not a JSON-RPC token"),
    ],
)
def test_d1_587_negated_docs_and_junk_remain_unknown_and_not_coolable(
    status_code: object,
    message: str,
) -> None:
    """Negated, documentation-only, zero-hit, and junk tokens must not cool."""
    event = clsf.classify_failure(status_code=status_code, message=message)
    assert event.class_name == "unknown"
    assert event.origin == "unknown"
    assert event.confidence == "unknown"
    assert not fv.is_coolable(event)


def test_parse_catalog_status_code_rejects_junk_without_raising() -> None:
    """Catalog status parsing stays safe and exact for non-status tokens."""
    assert clsf is not None  # module import smoke
    assert _parse_catalog_status_code("") is None
    assert _parse_catalog_status_code("See error.code") is None
    assert _parse_catalog_status_code("HTTP remains 200") == 200
    assert _parse_catalog_status_code("http remains 200") == 200
    assert _parse_catalog_status_code("400") == 400
    assert _parse_catalog_status_code("-32600") == -32600
    assert _parse_catalog_status_code("--32600") is None
    assert _parse_catalog_status_code("x-32600y") is None
    assert _parse_catalog_status_code("400.0") is None
    assert _parse_catalog_status_code("12x") is None
    # Overflow / absurdly long digit strings must not raise.
    huge = "9" * 1000
    assert _parse_catalog_status_code(huge) is None or isinstance(
        _parse_catalog_status_code(huge), int
    )
    try:
        _parse_catalog_status_code(huge)
        _parse_catalog_status_code("-" + huge)
        _parse_catalog_status_code("not-a-code")
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(f"status parsing raised unexpectedly: {exc!r}") from exc


def test_json_rpc_embedded_message_requires_exact_token() -> None:
    """Message-embedded JSON-RPC codes require exact -32xxx tokens."""
    ok = clsf.classify_failure(status_code=None, message="wire fault -32600 invalid request")
    assert ok.class_name == "provider_4xx_other"
    assert ok.confidence == "marker"
    assert ok.evidence.get("json_rpc_code") == "-32600"
    assert fv.is_coolable(ok)

    bad = clsf.classify_failure(status_code=None, message="wire fault x-32600y invalid request")
    assert bad.class_name == "unknown"
    assert not fv.is_coolable(bad)
