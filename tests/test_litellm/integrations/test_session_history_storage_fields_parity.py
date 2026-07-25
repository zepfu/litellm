"""Wave A4D golden-output parity and script-import-surface tests.

Pins CURRENT behavior of the two A4D target modules so the engineer's
behavior-preserving extraction can be verified by re-running these tests
post-move. Parity and script-import guards are GREEN before extraction;
structural ownership and packaging guards are intentionally RED until
the A4D implementation lands.

Target modules and their ORIGINAL line bands (per
.analysis/plan-godmodule-decomposition-r3-remediation-2026-07-23.md):

  aawm_session_history/backfill.py       :13803-14392 (16 functions)
  aawm_session_history/storage_fields.py :15473-16157 (26 functions)

The runtime golden fixture was captured from 5056b95a6e (post-A4C, pre-A4D)
by executing the live pre-move code with a type-preserving representation.
Each golden value is {"type": <python-type-name>, "value": <json-safe>}.
Tuples are {"type": "tuple", "value": [typed elements]}.
datetime/date use ISO-8601 strings. Decimal uses string.
Do not regenerate or modify the fixture after A4D lands.
"""

from __future__ import annotations

import ast
import inspect
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SCRIPTS_DIR = REPO_ROOT / "scripts"

_GOLDEN_PATH = FIXTURES_DIR / "a4d_parity_golden.json"


def _load_golden() -> Dict[str, Any]:
    with open(_GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Type-preserving comparison helpers
# ---------------------------------------------------------------------------


def _typed_repr(obj: Any) -> Dict[str, Any]:
    """Convert a Python object to the type-preserving golden schema."""
    if obj is None:
        return {"type": "NoneType", "value": None}
    if isinstance(obj, bool):
        return {"type": "bool", "value": obj}
    if isinstance(obj, int):
        return {"type": "int", "value": obj}
    if isinstance(obj, float):
        return {"type": "float", "value": obj}
    if isinstance(obj, str):
        return {"type": "str", "value": obj}
    if isinstance(obj, Decimal):
        return {"type": "Decimal", "value": str(obj)}
    if isinstance(obj, datetime):
        return {"type": "datetime", "value": obj.isoformat()}
    if isinstance(obj, date):
        return {"type": "date", "value": obj.isoformat()}
    if isinstance(obj, tuple):
        return {"type": "tuple", "value": [_typed_repr(x) for x in obj]}
    if isinstance(obj, list):
        return {"type": "list", "value": [_typed_repr(x) for x in obj]}
    if isinstance(obj, dict):
        return {"type": "dict", "value": {str(k): _typed_repr(v) for k, v in obj.items()}}
    return {"type": type(obj).__name__, "value": str(obj)}


def _assert_typed_equal(actual: Any, golden: Dict[str, Any], label: str = "") -> None:
    """Assert actual value matches the typed golden representation exactly."""
    actual_typed = _typed_repr(actual)
    assert actual_typed == golden, (
        f"typed mismatch{' at ' + label if label else ''}: "
        f"actual={actual_typed!r} != golden={golden!r}"
    )


# ---------------------------------------------------------------------------
# Nine scripts that consume storage mappers / backfill helpers
# ---------------------------------------------------------------------------

_NINE_SCRIPTS: List[str] = [
    "backfill_claude_auto_review_session_history.py",
    "backfill_local_cli_session_history.py",
    "backfill_provider_error_observations_from_docker_logs.py",
    "backfill_rate_limit_observations.py",
    "backfill_session_history.py",
    "backfill_session_history_latency.py",
    "backfill_session_history_runtime_identity.py",
    "repair_session_history_provider_cache.py",
    "repair_session_history_repository_identity.py",
]

# Expected import surface per script, keyed by source module.
# Derived by static analysis of ImportFrom statements in each script.
_SCRIPT_EXPECTED_IMPORTS: Dict[str, Dict[str, List[str]]] = {
    "backfill_claude_auto_review_session_history.py": {
        "litellm.integrations.aawm_agent_identity": [
            "_CLAUDE_AUTO_REVIEW_AGENT_NAME",
            "_CLAUDE_AUTO_REVIEW_LOGICAL_MODEL",
            "_CLAUDE_AUTO_REVIEW_TRACE_NAME",
            "_build_aawm_dsn",
            "_is_claude_permission_check_metadata",
            "_normalize_repository_identity",
        ],
        "litellm.integrations.aawm_session_history.identity_selection": [
            "select_first_identity",
        ],
    },
    "backfill_local_cli_session_history.py": {
        "litellm.integrations.aawm_agent_identity": [
            "_safe_float",
            "_safe_int",
            "_safe_str",
        ],
    },
    "backfill_provider_error_observations_from_docker_logs.py": {
        "litellm.integrations.aawm_agent_identity": [
            "_AAWM_PROVIDER_ERROR_OBSERVATION_INSERT_SQL",
            "_build_aawm_dsn",
            "_build_provider_error_observation_db_payload",
            "_classify_provider_error",
            "_ensure_session_history_schema",
        ],
    },
    "backfill_rate_limit_observations.py": {
        "litellm.integrations.aawm_agent_identity": [
            "_build_aawm_dsn",
            "_build_rate_limit_observations",
            "_ensure_session_history_schema",
            "_persist_session_history_records",
            "_rate_limit_storage_billing_period_end_at",
            "_rate_limit_storage_billing_period_start_at",
            "_rate_limit_storage_provider",
            "_rate_limit_storage_quota_limit",
            "_rate_limit_storage_quota_key",
            "_rate_limit_storage_quota_remaining",
            "_rate_limit_storage_quota_used",
            "_rate_limit_storage_remaining_pct",
        ],
    },
    "backfill_session_history.py": {
        "litellm.integrations.aawm_agent_identity": [
            "_build_aawm_dsn",
            "_build_session_history_record_from_langfuse_trace_observation",
            "_build_session_history_record_from_spend_log_row",
            "_derive_langfuse_trace_tags_from_spend_log_row",
            "_derive_request_tags_from_langfuse_metadata",
            "_enrich_backfill_anthropic_context_window_metadata",
            "_ensure_session_history_schema",
            "_extract_repository_identity_from_metadata_sources",
            "_extract_repository_identity_from_metadata_sources_with_source",
            "_extract_tenant_identity_from_metadata_sources",
            "_is_anthropic_session_history_context",
            "_persist_session_history_records",
            "_safe_float",
            "_safe_int",
        ],
    },
    "backfill_session_history_latency.py": {
        "litellm.integrations.aawm_agent_identity": [
            "_AAWM_SESSION_HISTORY_ALTER_STATEMENTS",
            "_AAWM_SESSION_HISTORY_INDEX_STATEMENTS",
            "_AAWM_SESSION_HISTORY_TABLE_SQL",
            "_SESSION_HISTORY_LATENCY_FIELDS",
            "_SESSION_HISTORY_PREVIOUS_GAP_FIELD",
            "_build_aawm_dsn",
            "_build_session_history_latency_breakdown",
            "_ensure_session_history_schema",
        ],
    },
    "backfill_session_history_runtime_identity.py": {
        "litellm.integrations.aawm_agent_identity": [
            "_build_session_runtime_identity",
        ],
    },
    "repair_session_history_provider_cache.py": {
        "litellm.integrations.aawm_agent_identity": [
            "_AAWM_SESSION_HISTORY_ALTER_STATEMENTS",
            "_AAWM_SESSION_HISTORY_INDEX_STATEMENTS",
            "_AAWM_SESSION_HISTORY_TABLE_SQL",
            "_AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INDEX_STATEMENTS",
            "_AAWM_SESSION_HISTORY_TOOL_ACTIVITY_TABLE_SQL",
            "_build_aawm_dsn",
            "_compute_provider_cache_miss_cost_state",
            "_ensure_session_history_schema",
            "_normalize_provider_cache_family",
            "_normalize_session_history_provider",
            "_resolve_provider_cache_state",
            "_safe_int",
        ],
    },
    "repair_session_history_repository_identity.py": {
        "litellm.integrations.aawm_agent_identity": [
            "_AAWM_REPOSITORY_AGENT_ID_RE",
            "_AAWM_REPOSITORY_AGENT_ROLE_VALUES",
            "_AAWM_REPOSITORY_PLACEHOLDER_VALUES",
            "_AAWM_REPOSITORY_WAVE_AGENT_RE",
            "_CODEX_MEMORY_REPOSITORY_SUFFIX",
            "_build_aawm_dsn",
            "_extract_repository_identity_from_metadata_sources",
            "_normalize_repository_identity",
        ],
        "litellm.integrations.aawm_session_history.identity_selection": [
            "select_first_identity",
        ],
    },
}


def _extract_script_imports(script_path: Path) -> Dict[str, List[str]]:
    """AST-parse a script and extract ImportFrom names grouped by module.

    Only captures imports from litellm.integrations.aawm_agent_identity
    and litellm.integrations.aawm_session_history* surfaces.
    """
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    result: Dict[str, List[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if not (
            module == "litellm.integrations.aawm_agent_identity"
            or module.startswith("litellm.integrations.aawm_session_history")
        ):
            continue
        names = sorted(alias.name for alias in node.names)
        if module in result:
            result[module] = sorted(set(result[module] + names))
        else:
            result[module] = names
    return result


# =========================================================================
# GREEN parity tests: storage-field golden output (typed)
# =========================================================================


def test_storage_fields_full_record_parity() -> None:
    """Storage-field mappers produce identical typed output for a full record."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()["sf_full"]
    record: Dict[str, Any] = {
        "provider": "openai", "model": "gpt-4o", "client_id": "client-abc",
        "quota_key": "openai:gpt-4o:daily", "quota_type": "rate_limit",
        "quota_limit": 10000.0, "quota_used": 4500.0, "quota_remaining": 5500.0,
        "remaining_pct": 55.0, "retry_after": 30,
        "billing_period_start_at": "2026-07-01T00:00:00Z",
        "billing_period_end_at": "2026-08-01T00:00:00Z",
        "observed_at": "2026-07-25T12:00:00Z", "status_code": 429,
    }
    _assert_typed_equal(pkg._rate_limit_storage_provider(record), golden["provider"], "provider")
    _assert_typed_equal(pkg._rate_limit_storage_client(record), golden["client"], "client")
    _assert_typed_equal(pkg._rate_limit_storage_quota_key(record), golden["quota_key"], "quota_key")
    _assert_typed_equal(pkg._rate_limit_storage_quota_type(record), golden["quota_type"], "quota_type")
    _assert_typed_equal(pkg._rate_limit_storage_remaining_pct(record), golden["remaining_pct"], "remaining_pct")
    _assert_typed_equal(pkg._rate_limit_storage_quota_limit(record), golden["quota_limit"], "quota_limit")
    _assert_typed_equal(pkg._rate_limit_storage_quota_used(record), golden["quota_used"], "quota_used")
    _assert_typed_equal(pkg._rate_limit_storage_quota_remaining(record), golden["quota_remaining"], "quota_remaining")
    _assert_typed_equal(pkg._rate_limit_storage_billing_period_start_at(record), golden["bp_start"], "bp_start")
    _assert_typed_equal(pkg._rate_limit_storage_billing_period_end_at(record), golden["bp_end"], "bp_end")


def test_storage_fields_minimal_record_parity() -> None:
    """Storage-field mappers produce identical typed output for a minimal record."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()["sf_minimal"]
    record: Dict[str, Any] = {"model": "claude-sonnet-4-20250514"}
    _assert_typed_equal(pkg._rate_limit_storage_provider(record), golden["provider"], "provider")
    _assert_typed_equal(pkg._rate_limit_storage_client(record), golden["client"], "client")
    _assert_typed_equal(pkg._rate_limit_storage_quota_key(record), golden["quota_key"], "quota_key")
    _assert_typed_equal(pkg._rate_limit_storage_quota_type(record), golden["quota_type"], "quota_type")
    _assert_typed_equal(pkg._rate_limit_storage_remaining_pct(record), golden["remaining_pct"], "remaining_pct")
    _assert_typed_equal(pkg._rate_limit_storage_quota_limit(record), golden["quota_limit"], "quota_limit")
    _assert_typed_equal(pkg._rate_limit_storage_quota_used(record), golden["quota_used"], "quota_used")
    _assert_typed_equal(pkg._rate_limit_storage_quota_remaining(record), golden["quota_remaining"], "quota_remaining")
    _assert_typed_equal(pkg._rate_limit_storage_billing_period_start_at(record), golden["bp_start"], "bp_start")
    _assert_typed_equal(pkg._rate_limit_storage_billing_period_end_at(record), golden["bp_end"], "bp_end")


def test_storage_fields_none_record_parity() -> None:
    """Storage-field mappers produce identical typed output for None-valued fields."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()["sf_none"]
    record: Dict[str, Any] = {
        k: None
        for k in (
            "provider", "model", "client_id", "quota_key", "quota_type",
            "quota_limit", "quota_used", "quota_remaining", "remaining_pct",
        )
    }
    _assert_typed_equal(pkg._rate_limit_storage_provider(record), golden["provider"], "provider")
    _assert_typed_equal(pkg._rate_limit_storage_client(record), golden["client"], "client")
    _assert_typed_equal(pkg._rate_limit_storage_quota_key(record), golden["quota_key"], "quota_key")
    _assert_typed_equal(pkg._rate_limit_storage_quota_type(record), golden["quota_type"], "quota_type")
    _assert_typed_equal(pkg._rate_limit_storage_remaining_pct(record), golden["remaining_pct"], "remaining_pct")
    _assert_typed_equal(pkg._rate_limit_storage_quota_limit(record), golden["quota_limit"], "quota_limit")
    _assert_typed_equal(pkg._rate_limit_storage_quota_used(record), golden["quota_used"], "quota_used")
    _assert_typed_equal(pkg._rate_limit_storage_quota_remaining(record), golden["quota_remaining"], "quota_remaining")


def test_numeric_detail_parity() -> None:
    """_rate_limit_storage_numeric_detail typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()["numeric_detail"]
    _assert_typed_equal(pkg._rate_limit_storage_numeric_detail({"v": 100}, "v"), golden["int"], "int")
    _assert_typed_equal(pkg._rate_limit_storage_numeric_detail({"v": 99.5}, "v"), golden["float"], "float")
    _assert_typed_equal(pkg._rate_limit_storage_numeric_detail({"v": "42"}, "v"), golden["str_num"], "str_num")
    _assert_typed_equal(pkg._rate_limit_storage_numeric_detail({}, "v"), golden["missing"], "missing")
    _assert_typed_equal(pkg._rate_limit_storage_numeric_detail({"v": None}, "v"), golden["none"], "none")


def test_timestamp_detail_parity() -> None:
    """_rate_limit_storage_timestamp_detail typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()["timestamp_detail"]
    _assert_typed_equal(
        pkg._rate_limit_storage_timestamp_detail({"ts": "2026-07-25T12:00:00Z"}, "ts"),
        golden["iso"], "iso",
    )
    _assert_typed_equal(pkg._rate_limit_storage_timestamp_detail({}, "ts"), golden["missing"], "missing")
    _assert_typed_equal(pkg._rate_limit_storage_timestamp_detail({"ts": None}, "ts"), golden["none"], "none")


# =========================================================================
# GREEN parity tests: DB payload tuples (typed, asserts tuple type)
# =========================================================================


def test_rate_limit_observation_db_payload_parity() -> None:
    """_build_rate_limit_observation_db_payload typed tuple parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()["rl_obs_payload"]
    record: Dict[str, Any] = {
        "provider": "openai", "model": "gpt-4o", "client_id": "client-abc",
        "quota_key": "openai:gpt-4o:daily", "quota_type": "rate_limit",
        "quota_limit": 10000.0, "quota_used": 4500.0, "quota_remaining": 5500.0,
        "remaining_pct": 55.0, "retry_after": 30,
        "billing_period_start_at": "2026-07-01T00:00:00Z",
        "billing_period_end_at": "2026-08-01T00:00:00Z",
        "observed_at": "2026-07-25T12:00:00Z", "status_code": 429,
    }
    result = pkg._build_rate_limit_observation_db_payload(record)
    assert isinstance(result, tuple), f"expected tuple, got {type(result).__name__}"
    _assert_typed_equal(result, golden, "rl_obs_payload")


def test_rate_limit_transition_db_payload_parity() -> None:
    """_build_rate_limit_transition_db_payload typed tuple parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()["rl_trans_payload"]
    record: Dict[str, Any] = {
        "transition_key": "openai:gpt-4o:daily:exhaustion",
        "limit_key": "openai:gpt-4o:daily",
        "provider": "openai", "client_family": "codex", "account_hash": "abc123",
        "transition_type": "exhaustion", "confidence": 0.95,
        "signals": [{"type": "remaining_pct_drop"}],
        "source": "rate_limit_callback",
        "old_observed_at": "2026-07-25T12:00:00Z", "new_observed_at": "2026-07-25T14:00:00Z",
        "old_provider_resets_at": None, "new_provider_resets_at": "2026-07-26T00:00:00Z",
        "old_used_percentage": 95.0, "new_used_percentage": 100.0,
        "old_remaining_requests": 500, "new_remaining_requests": 0,
        "old_used_requests": 9500, "new_used_requests": 10000,
        "old_total_requests": 10000, "new_total_requests": 10000,
        "inferred_window_start_at": None,
        "detection_window_start_at": "2026-07-25T12:00:00Z",
        "detection_window_end_at": "2026-07-25T14:00:00Z",
        "session_usage_summary": {"total_requests": 10000},
        "old_observation": {"remaining_pct": 5.0},
        "new_observation": {"remaining_pct": 0.0},
        "metadata": {"trigger": "automatic"},
    }
    result = pkg._build_rate_limit_transition_db_payload(record)
    assert isinstance(result, tuple), f"expected tuple, got {type(result).__name__}"
    _assert_typed_equal(result, golden, "rl_trans_payload")


def test_provider_error_observation_db_payload_parity() -> None:
    """_build_provider_error_observation_db_payload typed tuple parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()["prov_err_payload"]
    record: Dict[str, Any] = {
        "observed_at": "2026-07-25T15:00:00Z", "environment": "production",
        "provider": "openai", "model": "gpt-4o", "model_group": "gpt-4o-group",
        "route_family": "openai", "status_code": 429,
        "error_type": "rate_limit_error", "error_code": None,
        "error_class": "RateLimitError",
        "retry_after_seconds": 30, "expected_reset_at": None,
        "session_id": "sess-789", "trace_id": "trace-456", "litellm_call_id": "call-123",
        "metadata": {"source": "docker_logs"},
    }
    result = pkg._build_provider_error_observation_db_payload(record)
    assert isinstance(result, tuple), f"expected tuple, got {type(result).__name__}"
    _assert_typed_equal(result, golden, "prov_err_payload")


def test_alias_routing_audit_db_payload_parity() -> None:
    """_build_alias_routing_audit_db_payload typed tuple parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()["audit_payload"]
    record: Dict[str, Any] = {
        "provider": "openai", "model": "gpt-4o", "model_group": "gpt-4o-group",
        "repository": "org/repo", "session_id": "sess-audit",
        "trace_id": "trace-audit", "litellm_call_id": "call-audit",
        "metadata": {"requested_model_alias": "codex-auto", "codex_auto_agent_alias": "true"},
    }
    event: Dict[str, Any] = {
        "event_type": "cooldown_applied", "observed_at": "2026-07-25T16:00:00Z",
        "alias_model": "gpt-4o", "provider": "openai", "model": "gpt-4o",
        "lane_key": "openai:gpt-4o", "cooldown_key": "openai:gpt-4o:model",
        "attempt_number": 1, "cooldown_scope": "model", "cooldown_seconds": 60.0,
        "selected": True,
    }
    result = pkg._build_alias_routing_audit_db_payload(record, event, 0)
    assert isinstance(result, tuple), f"expected tuple, got {type(result).__name__}"
    _assert_typed_equal(result, golden, "audit_payload")


# =========================================================================
# GREEN parity tests: backfill helpers (typed)
# =========================================================================


def test_backfill_split_spend_log_parity() -> None:
    """_split_spend_log_proxy_server_request typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()
    result = pkg._split_spend_log_proxy_server_request(
        {"proxy_server_request": '{"method": "POST", "url": "/v1/chat/completions"}'}
    )
    _assert_typed_equal(result, golden["split_req"], "split_req")
    _assert_typed_equal(
        pkg._split_spend_log_proxy_server_request({}), golden["split_req_missing"], "split_req_missing"
    )


def test_backfill_trace_id_parity() -> None:
    """_extract_trace_id_from_spend_log_row typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()
    _assert_typed_equal(
        pkg._extract_trace_id_from_spend_log_row({"trace_id": "abc-123", "proxy_server_request": "{}"}),
        golden["trace_id"], "trace_id",
    )
    _assert_typed_equal(
        pkg._extract_trace_id_from_spend_log_row({}), golden["trace_id_missing"], "trace_id_missing"
    )


def test_backfill_session_id_parity() -> None:
    """_coerce_nested_session_id + _extract_session_id typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()
    _assert_typed_equal(pkg._coerce_nested_session_id("session-abc"), golden["session_id_str"], "str")
    _assert_typed_equal(pkg._coerce_nested_session_id({"session_id": "sess-123"}), golden["session_id_dict"], "dict")
    _assert_typed_equal(pkg._coerce_nested_session_id(None), golden["session_id_none"], "none")
    _assert_typed_equal(
        pkg._extract_session_id_from_spend_log_row({"session_id": "sess-direct", "proxy_server_request": "{}"}),
        golden["extract_session_id"], "extract_session_id",
    )


def test_backfill_tags_parity() -> None:
    """_coerce_spend_log_request_tags typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()
    _assert_typed_equal(pkg._coerce_spend_log_request_tags(["tag1", "tag2"]), golden["tags_list"], "list")
    _assert_typed_equal(pkg._coerce_spend_log_request_tags("single-tag"), golden["tags_str"], "str")
    _assert_typed_equal(pkg._coerce_spend_log_request_tags(None), golden["tags_none"], "none")


def test_backfill_searchable_text_parity() -> None:
    """_serialize_searchable_text typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()
    _assert_typed_equal(pkg._serialize_searchable_text("hello world"), golden["searchable_str"], "str")
    _assert_typed_equal(pkg._serialize_searchable_text({"key": "value"}), golden["searchable_dict"], "dict")
    _assert_typed_equal(pkg._serialize_searchable_text(None), golden["searchable_none"], "none")


def test_backfill_langfuse_session_id_parity() -> None:
    """_extract_langfuse_session_id typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()
    _assert_typed_equal(
        pkg._extract_langfuse_session_id({"session_id": "lf-session-1"}, {}),
        golden["lf_session_id"], "lf_session_id",
    )


def test_backfill_usage_from_observation_parity() -> None:
    """_build_usage_object_from_langfuse_observation typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()
    _assert_typed_equal(
        pkg._build_usage_object_from_langfuse_observation(
            {"usage": {"input": 100, "output": 50, "total": 150}, "model": "gpt-4o"}
        ),
        golden["usage_from_obs"], "usage_from_obs",
    )
    _assert_typed_equal(
        pkg._build_usage_object_from_langfuse_observation({}),
        golden["usage_from_obs_empty"], "usage_from_obs_empty",
    )


def test_backfill_first_response_message_parity() -> None:
    """_extract_first_langfuse_response_message typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()
    _assert_typed_equal(
        pkg._extract_first_langfuse_response_message(
            {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}
        ),
        golden["first_resp_msg"], "first_resp_msg",
    )
    _assert_typed_equal(
        pkg._extract_first_langfuse_response_message(None),
        golden["first_resp_msg_none"], "first_resp_msg_none",
    )


def test_backfill_infer_provider_parity() -> None:
    """_infer_provider_from_langfuse_observation typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()
    _assert_typed_equal(
        pkg._infer_provider_from_langfuse_observation({"model": "gpt-4o"}, {}),
        golden["infer_prov_openai"], "openai",
    )
    _assert_typed_equal(
        pkg._infer_provider_from_langfuse_observation({"model": "claude-sonnet-4-20250514"}, {}),
        golden["infer_prov_anthropic"], "anthropic",
    )


def test_backfill_derive_tags_parity() -> None:
    """_derive_request_tags_from_langfuse_metadata typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()
    _assert_typed_equal(
        pkg._derive_request_tags_from_langfuse_metadata({"tags": ["production", "v2"]}),
        golden["derive_tags"], "derive_tags",
    )
    _assert_typed_equal(
        pkg._derive_request_tags_from_langfuse_metadata({}),
        golden["derive_tags_empty"], "derive_tags_empty",
    )


def test_backfill_derive_trace_tags_spend_log_parity() -> None:
    """_derive_langfuse_trace_tags_from_spend_log_row typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()
    _assert_typed_equal(
        pkg._derive_langfuse_trace_tags_from_spend_log_row({"request_tags": ["tag-a", "tag-b"]}),
        golden["derive_trace_tags_spend"], "derive_trace_tags_spend",
    )


def test_backfill_agent_context_parity() -> None:
    """_extract_agent_context_from_langfuse_trace_observation typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()
    _assert_typed_equal(
        pkg._extract_agent_context_from_langfuse_trace_observation(
            {"input": [{"role": "system", "content": "You are a 'worker' agent."}]}, {}
        ),
        golden["agent_context"], "agent_context",
    )


# =========================================================================
# GREEN parity tests: alias routing audit helpers (typed)
# =========================================================================


def test_extract_alias_routing_audit_events_parity() -> None:
    """_extract_alias_routing_audit_events typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()
    _assert_typed_equal(
        pkg._extract_alias_routing_audit_events({}),
        golden["extract_audit_events_empty"], "empty",
    )
    _assert_typed_equal(
        pkg._extract_alias_routing_audit_events(
            {"metadata": {"aawm_alias_routing_audit_events": [{"event_type": "cooldown", "observed_at": "2026-07-25T10:00:00Z"}]}}
        ),
        golden["extract_audit_events_with"], "with",
    )


def test_infer_alias_routing_family_parity() -> None:
    """_infer_alias_routing_family typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()["infer_family"]
    event: Dict[str, Any] = {
        "event_type": "cooldown_applied", "observed_at": "2026-07-25T16:00:00Z",
        "alias_model": "gpt-4o", "provider": "openai", "model": "gpt-4o",
    }
    metadata: Dict[str, Any] = {"requested_model_alias": "codex-auto", "codex_auto_agent_alias": "true"}
    _assert_typed_equal(pkg._infer_alias_routing_family(event, metadata), golden["codex"], "codex")
    _assert_typed_equal(pkg._infer_alias_routing_family({}, {}), golden["empty"], "empty")
    _assert_typed_equal(
        pkg._infer_alias_routing_family({}, {"anthropic_auto_agent_alias": "true"}),
        golden["anthropic"], "anthropic",
    )


def test_rate_limit_previous_observation_row_to_dict_parity() -> None:
    """_rate_limit_previous_observation_row_to_dict typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()

    class FakeRow:
        def __init__(self, d: Dict[str, Any]) -> None:
            self._d = d

        def keys(self) -> Any:
            return self._d.keys()

        def __getitem__(self, k: str) -> Any:
            return self._d[k]

    row = FakeRow({"provider": "openai", "remaining_pct": 55.0})
    _assert_typed_equal(pkg._rate_limit_previous_observation_row_to_dict(row), golden["row_to_dict"], "row")
    _assert_typed_equal(pkg._rate_limit_previous_observation_row_to_dict(None), golden["row_to_dict_none"], "none")


def test_rate_limit_observation_only_requested_parity() -> None:
    """_rate_limit_observation_only_requested typed golden parity."""
    import litellm.integrations.aawm_agent_identity as pkg

    golden = _load_golden()
    _assert_typed_equal(
        pkg._rate_limit_observation_only_requested({"metadata": {"aawm_rate_limit_observation_only": True}}),
        golden["obs_only_true"], "true",
    )
    _assert_typed_equal(
        pkg._rate_limit_observation_only_requested({"metadata": {"other": "v"}}),
        golden["obs_only_false"], "false",
    )
    _assert_typed_equal(
        pkg._rate_limit_observation_only_requested({}),
        golden["obs_only_missing"], "missing",
    )


# =========================================================================
# GREEN: script import surface tests (AST-verified)
# =========================================================================


def test_all_nine_scripts_exist() -> None:
    """All nine backfill/repair scripts must exist."""
    missing = [s for s in _NINE_SCRIPTS if not (SCRIPTS_DIR / s).is_file()]
    assert not missing, f"expected scripts to exist: {missing}"


def test_script_import_surfaces_match_ast() -> None:
    """AST-parse each script's ImportFrom statements and assert the exact
    imported-name set equals the pinned expected set per script."""
    mismatches: List[str] = []
    for script_name, expected_by_module in _SCRIPT_EXPECTED_IMPORTS.items():
        script_path = SCRIPTS_DIR / script_name
        if not script_path.is_file():
            mismatches.append(f"{script_name}: file missing")
            continue
        actual = _extract_script_imports(script_path)
        for module, expected_names in expected_by_module.items():
            actual_names = actual.get(module, [])
            if sorted(actual_names) != sorted(expected_names):
                mismatches.append(
                    f"{script_name} from {module}: "
                    f"expected={sorted(expected_names)} actual={sorted(actual_names)}"
                )
        # Check for unexpected modules
        for module in actual:
            if module not in expected_by_module:
                mismatches.append(
                    f"{script_name}: unexpected import module {module} "
                    f"with names {actual[module]}"
                )
    assert not mismatches, "script import surface mismatches:\n" + "\n".join(mismatches)


def test_script_import_surfaces_resolve() -> None:
    """Every symbol each script imports must be resolvable via getattr on
    the appropriate package (facade compatibility)."""
    import litellm.integrations.aawm_agent_identity as pkg

    failures: List[Tuple[str, str, str]] = []
    for script_name, expected_by_module in _SCRIPT_EXPECTED_IMPORTS.items():
        for module, names in expected_by_module.items():
            if module == "litellm.integrations.aawm_agent_identity":
                target = pkg
            else:
                # aawm_session_history or submodules
                try:
                    target = __import__(module, fromlist=[""])
                except ImportError:
                    failures.append((script_name, module, "<module not importable>"))
                    continue
            for sym in names:
                if getattr(target, sym, None) is None:
                    failures.append((script_name, module, sym))
    assert not failures, f"script import surface symbols unresolvable: {failures}"


def test_script_import_surface_a4d_coverage() -> None:
    """A4D storage mapper symbols used by scripts must be in the surface."""
    a4d_storage_in_scripts = {
        "_rate_limit_storage_billing_period_end_at",
        "_rate_limit_storage_billing_period_start_at",
        "_rate_limit_storage_provider",
        "_rate_limit_storage_quota_limit",
        "_rate_limit_storage_quota_key",
        "_rate_limit_storage_quota_remaining",
        "_rate_limit_storage_quota_used",
        "_rate_limit_storage_remaining_pct",
        "_build_provider_error_observation_db_payload",
    }
    all_syms: set = set()
    for expected_by_module in _SCRIPT_EXPECTED_IMPORTS.values():
        for names in expected_by_module.values():
            all_syms.update(names)
    assert a4d_storage_in_scripts <= all_syms, (
        f"A4D storage symbols missing from script surfaces: "
        f"{a4d_storage_in_scripts - all_syms}"
    )


# =========================================================================
# GREEN: async fetch/derive helper signature pinning
# =========================================================================


def test_async_helpers_are_coroutine_functions() -> None:
    """The four async helpers must be coroutine functions (not sync)."""
    import litellm.integrations.aawm_agent_identity as pkg

    async_names = [
        "_fetch_previous_rate_limit_observation",
        "_fetch_previous_rate_limit_observations",
        "_derive_rate_limit_transitions",
        "_filter_meaningful_rate_limit_observations",
    ]
    non_async = [
        name
        for name in async_names
        if not inspect.iscoroutinefunction(getattr(pkg, name, None))
    ]
    assert not non_async, f"expected coroutine functions: {non_async}"
