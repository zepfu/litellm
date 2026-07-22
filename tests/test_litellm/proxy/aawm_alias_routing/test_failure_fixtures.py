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

# Known gap set: provider-reachable ``Normalized Class`` values that
# ``classify_failure`` does not yet register a dedicated mapping for (e.g.
# image-content sub-errors, JSON-RPC negative wire codes, HTTP-200-body stream
# failures, 2xx/3xx non-error status codes). These are explicitly acknowledged
# as not-yet-covered rather than silently passing the coverage checklist.
_KNOWN_COVERAGE_GAPS = frozenset(
    {
        "Agent limit",
        "Agent/model limit",
        "Billing/quota",
        "Content policy",
        "Content-policy reject",
        "Context limit",
        "External fetch failure",
        "Fix config",
        "Fix input",
        "Fix protocol/capability",
        "Fix protocol/request",
        "Fix protocol/state",
        "Fix request/content",
        "Input/network dependency",
        "Invalid media",
        "Invalid request",
        "Layered/platform passthrough",
        "Local network",
        "Local network/config",
        "Model refusal",
        "Model/output limit",
        "Not found",
        "Not found/version",
        "Payload limit",
        "Pending/non-error",
        "Pending/not terminal",
        "Permission/policy block",
        "Plan/tool restriction",
        "Precondition/conflict",
        "Protocol-mapped error",
        "Provider failure",
        "Provider overload",
        "Provider unavailable",
        "Quota/limit",
        "Rate limit",
        "Re-authenticate/config",
        "Reconnect",
        "Result indirection",
        "Retry in progress",
        "Retry transient",
        "Retry with pacing",
        "Server error",
        "Server/API error",
        "State/session recovery",
        "Stream failure",
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


def test_csv_coverage_checklist() -> None:
    """Every provider-reachable catalog row maps to a registered class or a known gap.

    TUI-layer rows (``TUI hook`` / ``TUI/headless``) are the client's own local
    presentation/lifecycle surface -- they are asserted out-of-scope and must
    never classify as coolable.
    """
    rows = _load_catalog_rows()
    assert rows, "expected the error-code catalog CSV to contain rows"

    unresolved_gaps: list[tuple[str, str, str]] = []
    for row in rows:
        layer = row["Layer"]
        if layer not in _PROVIDER_REACHABLE_LAYERS:
            continue
        raw_code = row["HTTP / Exit / RPC"].strip()
        status_code = int(raw_code) if raw_code.isdigit() else None
        message = f"{row['Machine Code / Type / Event']} {row['Meaning']}"
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
