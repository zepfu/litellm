"""Focused RR-006 regression tests for aawm_agent_identity helpers."""

from __future__ import annotations

from pathlib import Path

import queue
from typing import Any

import pytest

from litellm.integrations import aawm_agent_identity


def test_rr006_normalize_provider_cache_family_o4_and_gemma() -> None:
    assert (
        aawm_agent_identity._normalize_provider_cache_family(None, "o4-mini")
        == "openai"
    )
    assert (
        aawm_agent_identity._normalize_provider_cache_family(None, "google/gemma-2-9b")
        == "gemini"
    )
    assert (
        aawm_agent_identity._normalize_provider_cache_family(None, "gemma-2-9b")
        == "gemini"
    )


def test_rr006_infer_model_family_does_not_treat_project_or_gpt_pro_as_gemini() -> None:
    family, tier = aawm_agent_identity._infer_model_family_and_tier(
        "gpt-5-pro",
        "project-prod-workspace",
    )
    assert family == "openai"
    assert tier == "pro"

    family, tier = aawm_agent_identity._infer_model_family_and_tier(
        "gemini-1.5-pro",
    )
    assert family == "gemini"
    assert tier == "pro"

    family, tier = aawm_agent_identity._infer_model_family_and_tier(
        "repository project",
    )
    assert family is None
    assert tier is None


def test_rr006_payload_codex_memory_markers_depth_and_cycle_safe() -> None:
    required = aawm_agent_identity._CODEX_MEMORY_WORKFLOW_REQUIRED_MARKER
    context = aawm_agent_identity._CODEX_MEMORY_WORKFLOW_CONTEXT_MARKERS[0]

    cyclic: dict[str, Any] = {"text": f"{required} {context}"}
    cyclic["self"] = cyclic
    # Must not hang / RecursionError on cyclic graphs.
    assert (
        aawm_agent_identity._payload_contains_codex_memory_workflow_markers(cyclic)
        is True
    )

    deep: dict[str, Any] = {}
    current = deep
    for _ in range(40):
        current["child"] = {}
        current = current["child"]
    current["text"] = f"{required} {context}"
    # Depth-capped traversal should not recurse forever and should fail closed
    # past the guard rather than crash.
    assert (
        aawm_agent_identity._payload_contains_codex_memory_workflow_markers(deep)
        is False
    )


def test_rr006_ensure_mutable_headers_reattaches_missing_headers() -> None:
    kwargs: dict[str, Any] = {}
    headers = aawm_agent_identity._ensure_mutable_headers(kwargs)
    headers["langfuse_trace_user_id"] = "repo-x"
    assert (
        kwargs["litellm_params"]["proxy_server_request"]["headers"][
            "langfuse_trace_user_id"
        ]
        == "repo-x"
    )


def test_rr006_provider_error_fingerprint_excludes_raw_volatility() -> None:
    meta_a = {
        "upstream_provider_name": "openai",
        "upstream_error_raw": "request_id=aaa Authorization: Bearer sk-live-1",
    }
    meta_b = {
        "upstream_provider_name": "openai",
        "upstream_error_raw": "request_id=bbb Authorization: Bearer sk-live-2",
    }
    fp_a = aawm_agent_identity._build_provider_error_fingerprint(
        provider="openai",
        model="gpt-4o",
        model_group=None,
        status_code=429,
        error_code="rate_limit",
        error_type="RateLimitError",
        error_class="rate_limit",
        observation_metadata=meta_a,
    )
    fp_b = aawm_agent_identity._build_provider_error_fingerprint(
        provider="openai",
        model="gpt-4o",
        model_group=None,
        status_code=429,
        error_code="rate_limit",
        error_type="RateLimitError",
        error_class="rate_limit",
        observation_metadata=meta_b,
    )
    assert fp_a == fp_b


def test_rr006_redacts_upstream_error_raw_secrets() -> None:
    observation_metadata: dict[str, Any] = {}
    aawm_agent_identity._enrich_provider_error_observation_metadata(
        observation_metadata=observation_metadata,
        dicts=[{"raw": "failed Authorization: Bearer sk-super-secret-value"}],
        error_text="failed",
    )
    raw = observation_metadata["upstream_error_raw"]
    assert "sk-super-secret-value" not in raw
    assert "[REDACTED]" in raw


def test_rr006_quota_period_from_window_minutes_hourly_round_trip() -> None:
    assert aawm_agent_identity._quota_period_from_window_minutes(60) == "hourly"
    assert aawm_agent_identity._window_minutes_from_quota_period("hourly") == 60


def test_rr006_json_safe_rate_limit_value_allows_aliased_non_cyclic_refs() -> None:
    shared = {"k": 1}
    value = {"a": shared, "b": shared}
    safe = aawm_agent_identity._json_safe_rate_limit_value(value)
    assert safe == {"a": {"k": 1}, "b": {"k": 1}}


def test_rr006_coerce_rate_limit_payload_rejects_oversized_literal_eval_input() -> None:
    oversized = "[" * 9000 + "]" * 9000
    assert aawm_agent_identity._coerce_rate_limit_payload(oversized) is None
    assert aawm_agent_identity._coerce_rate_limit_payload("{'ok': 1}") == {"ok": 1}


def test_rr006_content_to_text_skips_non_text_blocks() -> None:
    text = aawm_agent_identity._content_to_text(
        [
            {"type": "tool_use", "name": "bash"},
            {"type": "text", "text": "You are 'worker' and you are working"},
            {"type": "thinking", "thinking": "secret"},
        ]
    )
    assert text == "You are 'worker' and you are working"
    assert "tool_use" not in text


def test_rr006_is_codex_subagent_false_for_falsey_subagent_flag() -> None:
    kwargs = {
        "litellm_params": {
            "metadata": {"source": {"subagent": False}},
        }
    }
    assert aawm_agent_identity._is_codex_subagent_context(kwargs) is False
    kwargs["litellm_params"]["metadata"]["source"]["subagent"] = True
    assert aawm_agent_identity._is_codex_subagent_context(kwargs) is True


def test_rr006_apply_claude_auto_review_identity_does_not_null_existing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        aawm_agent_identity,
        "_is_claude_permission_check_metadata",
        lambda _metadata: True,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_extract_claude_auto_review_source_model",
        lambda _metadata, _model: None,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_extract_claude_project_from_metadata_tags",
        lambda _metadata: None,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_apply_claude_auto_review_metadata",
        lambda metadata, **_kwargs: None,
    )
    record = {
        "repository": "keep-me",
        "tenant_id": "tenant-keep",
        "metadata": {"claude_permission_check": True},
        "model": "claude-sonnet-4-6",
    }
    # Force normalize paths to None while original raw values remain.
    monkeypatch.setattr(
        aawm_agent_identity,
        "_normalize_repository_identity",
        lambda _value: None,
    )
    aawm_agent_identity._apply_claude_auto_review_identity_to_record(record)
    assert record["repository"] == "keep-me"
    assert record["tenant_id"] == "tenant-keep"


def test_rr006_build_usage_object_from_metadata_merges_input_token_details() -> None:
    usage = aawm_agent_identity._build_usage_object_from_metadata(
        {
            "usage_object": {
                "input_tokens_details": {"audio_tokens": 3},
            },
            "usage_cache_read_input_tokens": 11,
        }
    )
    assert usage is not None
    assert usage["input_tokens_details"]["audio_tokens"] == 3
    assert usage["input_tokens_details"]["cached_tokens"] == 11


def test_rr006_token_count_payload_ignores_bare_total_without_token_siblings() -> None:
    assert (
        aawm_agent_identity._build_usage_object_from_token_count_payload(
            {"total": 99, "page": 1}
        )
        is None
    )
    usage = aawm_agent_identity._build_usage_object_from_token_count_payload(
        {"input_tokens": 2, "output_tokens": 3, "total": 5}
    )
    assert usage is not None
    assert usage["total_tokens"] == 5


def test_rr006_lookup_bundled_model_cost_prefers_provider_qualified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_cost = {
        "shared-model": {"input_cost_per_token": 1.0, "litellm_provider": "other"},
        "anthropic/shared-model": {
            "input_cost_per_token": 2.0,
            "litellm_provider": "anthropic",
        },
    }
    monkeypatch.setattr(
        aawm_agent_identity,
        "_load_bundled_model_cost_map",
        lambda: model_cost,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_bundled_model_cost_casefold_lookup",
        lambda: {key.lower(): key for key in model_cost},
    )
    info = aawm_agent_identity._lookup_bundled_model_cost_info(
        model="shared-model",
        custom_llm_provider="anthropic",
    )
    assert info is not None
    assert info["input_cost_per_token"] == 2.0


def test_rr006_normalize_agent_id_copies_metadata_before_mutate() -> None:
    agent_id = "01234567-89ab-cdef-0123-456789abcdef"
    shared_metadata = {"agent_id": agent_id}
    original = {"metadata": shared_metadata, "agent_id": agent_id}
    shallow = dict(original)
    aawm_agent_identity._normalize_agent_id_on_record(shallow)
    # Metadata object must not be shared with the shallow-copied caller's dict.
    assert shallow["metadata"] is not shared_metadata
    assert shallow["agent_id"] == agent_id
    shallow["metadata"]["probe"] = True
    assert "probe" not in shared_metadata


def test_rr006_safe_str_shared_helper() -> None:
    assert aawm_agent_identity._safe_str("  abc  ") == "abc"
    assert aawm_agent_identity._safe_str("") is None
    assert aawm_agent_identity._safe_str(None) is None


def test_rr006_build_session_runtime_identity_no_dead_elif_branch() -> None:
    identity = aawm_agent_identity._build_session_runtime_identity(
        metadata={"cc_version": "1.2.3"},
        allow_runtime=False,
    )
    assert identity.get("client_name") == "claude-code"
    assert identity.get("client_version") == "1.2.3"


def test_rr006_xai_quota_period_monthly_only_for_billing_reset_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_at = "2026-07-17T00:00:00Z"

    def _fake_context(*_args, **_kwargs):
        return {
            "provider": "xai",
            "observed_at": observed_at,
            "model": "grok-3",
            "metadata": {"xai_oauth_public_model": "grok-3"},
        }

    monkeypatch.setattr(
        aawm_agent_identity,
        "_build_rate_limit_context",
        _fake_context,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_looks_like_xai_oauth_rate_limit_context",
        lambda _context: True,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_extract_xai_oauth_account_hash",
        lambda _metadata: "acct",
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_iter_rate_limit_dicts",
        lambda *_roots: [
            {
                "source": "xai_oauth_response_headers",
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "90",
                "x-ratelimit-reset-requests": "30",
            }
        ],
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_resolve_rate_limit_reset_at",
        lambda *_args, **_kwargs: ("2026-07-17T00:00:30Z", False),
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_finalize_rate_limit_observation",
        lambda observation, _context: observation,
    )

    observations = aawm_agent_identity._extract_xai_oauth_header_rate_limit_observations(
        {},
        None,
        observed_at,
    )
    assert observations
    assert observations[0]["quota_period"] is None


@pytest.mark.asyncio
async def test_rr006_session_history_pool_uses_server_settings_without_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_pool = object()
    create_pool_calls: list[dict] = []

    class FakeAsyncpg:
        async def create_pool(self, **kwargs):
            create_pool_calls.append(kwargs)
            return created_pool

    monkeypatch.setattr(aawm_agent_identity, "_aawm_session_history_pools", {})
    monkeypatch.setattr(
        aawm_agent_identity, "_build_aawm_dsn", lambda: "postgresql://aawm@test/db"
    )
    monkeypatch.setattr(aawm_agent_identity, "_get_session_history_pool_max_size", lambda: 3)
    monkeypatch.setattr(
        aawm_agent_identity,
        "_get_session_history_command_timeout_seconds",
        lambda: 42.0,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_get_session_history_statement_cache_size",
        lambda: 0,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_get_session_history_application_name",
        lambda: "aawm-litellm-test",
    )
    monkeypatch.setattr(
        aawm_agent_identity.importlib,
        "import_module",
        lambda name: FakeAsyncpg() if name == "asyncpg" else None,
    )

    pool = await aawm_agent_identity._get_aawm_session_history_pool()
    assert pool is created_pool
    assert len(create_pool_calls) == 1
    assert "init" not in create_pool_calls[0]
    assert create_pool_calls[0]["server_settings"] == {
        "application_name": "aawm-litellm-test"
    }


@pytest.mark.asyncio
async def test_rr006_rate_limit_side_write_failure_is_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    records = [
        {
            "litellm_call_id": "call-side-write-best-effort",
            "_skip_session_history": True,
            "rate_limit_observations": [
                {
                    "observed_at": "2026-07-17T00:00:00Z",
                    "source": "test",
                    "provider": "anthropic",
                    "limit_key": "k",
                }
            ],
        }
    ]

    class Conn:
        async def executemany(self, *_args, **_kwargs):
            raise RuntimeError("rate-limit insert unavailable")

    async def empty_openrouter(*_a, **_k):
        return []

    async def fake_filter(_conn, observations):
        return observations, {}

    async def fake_transitions(_conn, observations, _previous):
        return []

    monkeypatch.setattr(
        aawm_agent_identity,
        "_build_openrouter_free_daily_observations_for_records",
        empty_openrouter,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_filter_meaningful_rate_limit_observations",
        fake_filter,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_derive_rate_limit_transitions",
        fake_transitions,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_build_rate_limit_observation_db_payload",
        lambda observation: (observation,),
    )
    warning_mock = MagicMock()
    monkeypatch.setattr(aawm_agent_identity.verbose_logger, "warning", warning_mock)

    await aawm_agent_identity._persist_rate_limit_observations_best_effort(
        Conn(),
        records,
        history_records=[],
    )
    assert warning_mock.called


def test_rr006_flush_retry_exhaustion_without_spool_does_not_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    attempts: list[int] = []

    def fake_flush(_batch, **kwargs):
        attempts.append(1)
        failure_callback = kwargs.get("failure_callback")
        if failure_callback is not None:
            failure_callback(OSError("pgbouncer unavailable"))
        return False

    def failing_spool(*_args, **_kwargs):
        raise OSError("spool unwritable")

    monkeypatch.setattr(
        aawm_agent_identity,
        "get_secret_str",
        lambda key: "0"
        if key == "AAWM_SESSION_HISTORY_FAILED_FLUSH_MAX_RETRIES"
        else None,
    )
    monkeypatch.setattr(aawm_agent_identity, "_flush_session_history_batch", fake_flush)
    monkeypatch.setattr(
        aawm_agent_identity, "_spool_session_history_records", failing_spool
    )
    monkeypatch.setattr(aawm_agent_identity.time, "sleep", lambda _seconds: None)
    error_mock = MagicMock()
    monkeypatch.setattr(aawm_agent_identity.verbose_logger, "error", error_mock)
    monkeypatch.setattr(aawm_agent_identity.verbose_logger, "warning", MagicMock())
    monkeypatch.setattr(aawm_agent_identity.verbose_logger, "exception", MagicMock())

    aawm_agent_identity._flush_session_history_batch_with_retry(
        [{"litellm_call_id": "call-drop"}]
    )
    assert len(attempts) == 1
    assert error_mock.called


def test_rr006_shutdown_drains_queue_to_spool_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    class FakeWorker:
        def __init__(self) -> None:
            self.join_timeout = None

        def join(self, timeout):
            self.join_timeout = timeout

    class FakeQueue:
        def __init__(self) -> None:
            self.put_calls = []
            self.items = [
                {"litellm_call_id": "queued-1"},
                {"litellm_call_id": "queued-2"},
            ]
            self.maxsize = 1024

        def put(self, item, timeout):
            self.put_calls.append((item, timeout))

        def get_nowait(self):
            if not self.items:
                raise queue.Empty
            return self.items.pop(0)


    worker = FakeWorker()
    q = FakeQueue()
    spooled = []

    monkeypatch.setattr(aawm_agent_identity, "_aawm_session_history_worker", worker)
    monkeypatch.setattr(aawm_agent_identity, "_aawm_session_history_queue", q)
    monkeypatch.setattr(
        aawm_agent_identity,
        "_get_session_history_flush_interval_seconds",
        lambda: 0.5,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_spool_session_history_records",
        lambda records, **kwargs: spooled.append((list(records), kwargs)),
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_session_history_queue_depth_summary",
        lambda: "queue_depth=0",
    )
    monkeypatch.setattr(aawm_agent_identity.verbose_logger, "warning", MagicMock())
    monkeypatch.setattr(aawm_agent_identity.verbose_logger, "exception", MagicMock())

    aawm_agent_identity._shutdown_session_history_worker()

    assert q.put_calls == [(None, 0.5)]
    assert worker.join_timeout == 1.0
    assert len(spooled) == 1
    assert [r["litellm_call_id"] for r in spooled[0][0]] == ["queued-1", "queued-2"]
    assert spooled[0][1]["reason"] == "shutdown post-join drain"


def test_rr006_embedded_json_parse_attempt_budget() -> None:
    # Many '{' with no valid objects would previously pay unbounded raw_decode.
    noisy = "{" + "{x" * 200
    assert aawm_agent_identity._extract_embedded_json_payload_dicts(noisy) == []
    # Still finds a real payload within the attempt budget.
    payload = '{"code": "rate_limit", "message": "slow down"}'
    mixed = ("{x" * 10) + payload
    dicts = aawm_agent_identity._extract_embedded_json_payload_dicts(mixed)
    assert any(d.get("code") == "rate_limit" for d in dicts)


def test_rr006_rate_limit_header_map_used_once_per_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"map": 0}
    original = aawm_agent_identity._rate_limit_header_map

    def counting_map(candidate):
        calls["map"] += 1
        return original(candidate)

    monkeypatch.setattr(aawm_agent_identity, "_rate_limit_header_map", counting_map)
    candidate = {
        "x-ratelimit-limit-requests": "10",
        "x-ratelimit-remaining-requests": "9",
        "x-ratelimit-reset-requests": "30",
    }
    lower = aawm_agent_identity._rate_limit_header_map(candidate)
    # Repeated lookups with explicit lower_headers should not remap.
    before = calls["map"]
    for _ in range(5):
        aawm_agent_identity._get_rate_limit_header_value(
            candidate,
            "x-ratelimit-limit-requests",
            lower_headers=lower,
        )
    assert calls["map"] == before


def test_rr006_lazy_litellm_helpers_cache_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aawm_agent_identity._litellm_module = None
    aawm_agent_identity._response_api_logging_utils = None
    aawm_agent_identity._response_api_logging_utils_loaded = False

    first = aawm_agent_identity._get_litellm_module()
    second = aawm_agent_identity._get_litellm_module()
    assert first is second
    # Response utils may be present; ensure loader is sticky.
    utils = aawm_agent_identity._get_response_api_logging_utils()
    assert aawm_agent_identity._response_api_logging_utils_loaded is True
    assert aawm_agent_identity._get_response_api_logging_utils() is utils


def test_rr006_repository_source_trust_helper_dedupe() -> None:
    general_ok = "metadata.repository"
    # codex memory + metadata is trusted by both
    memory = "foo.metadata.bar.codex_memory_workflow"
    assert aawm_agent_identity._is_repository_source_trusted_for_tenant(memory) is True
    assert (
        aawm_agent_identity._is_repository_source_trusted_for_codex_tenant(memory)
        is True
    )
    # general metadata marker only trusted by general helper
    assert (
        aawm_agent_identity._is_repository_source_trusted_for_tenant(
            "payload.metadata.repository"
        )
        is True
    )
    assert (
        aawm_agent_identity._is_repository_source_trusted_for_codex_tenant(
            "payload.metadata.repository"
        )
        is False
    )
    # route rollup only general
    assert (
        aawm_agent_identity._is_repository_source_trusted_for_tenant(
            "x.aawm_route_rollup_context.group_header_label"
        )
        is True
    )
    assert (
        aawm_agent_identity._is_repository_source_trusted_for_codex_tenant(
            "x.aawm_route_rollup_context.group_header_label"
        )
        is False
    )
    _ = general_ok


@pytest.mark.asyncio
async def test_rr006_primary_persist_uses_transaction_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    entered = {"count": 0}

    class TrackingTx:
        async def __aenter__(self):
            entered["count"] += 1
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class Conn:
        def __init__(self):
            self.execute = AsyncMock()
            self.executemany = AsyncMock()

        def transaction(self):
            return TrackingTx()

    conn = Conn()

    class Pool:
        def acquire(self):
            class CM:
                async def __aenter__(self_inner):
                    return conn

                async def __aexit__(self_inner, *args):
                    return False

            return CM()

    monkeypatch.setattr(
        aawm_agent_identity,
        "_get_aawm_session_history_pool",
        AsyncMock(return_value=Pool()),
    )
    monkeypatch.setattr(
        aawm_agent_identity, "_ensure_session_history_schema", AsyncMock()
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_apply_claude_auto_review_parent_identity_from_store",
        AsyncMock(),
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_build_session_history_db_payload",
        lambda record: ("call-tx",),
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_build_tool_activity_db_payloads",
        lambda record: [],
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_update_session_history_previous_gap_ms",
        AsyncMock(),
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_persist_tool_definition_snapshots_best_effort",
        AsyncMock(),
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_persist_rate_limit_observations_best_effort",
        AsyncMock(),
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_persist_provider_error_observations_best_effort",
        AsyncMock(),
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_persist_alias_routing_audit_best_effort",
        AsyncMock(),
    )

    await aawm_agent_identity._persist_session_history_record(
        {"litellm_call_id": "call-tx"}
    )
    assert entered["count"] == 1
    conn.execute.assert_awaited()


def test_rr006_session_history_sql_module_exports_insert_and_table_sql() -> None:
    from litellm.integrations import aawm_session_history_sql as sql_mod
    from litellm.integrations import aawm_agent_identity as identity

    assert "CREATE TABLE IF NOT EXISTS public.session_history" in (
        sql_mod._AAWM_SESSION_HISTORY_TABLE_SQL
    )
    assert "INSERT INTO public.session_history" in (
        sql_mod._AAWM_SESSION_HISTORY_INSERT_SQL
    )
    # Compatibility re-export from the identity callback module.
    assert (
        identity._AAWM_SESSION_HISTORY_INSERT_SQL
        is sql_mod._AAWM_SESSION_HISTORY_INSERT_SQL
    )
    assert (
        identity._SESSION_HISTORY_PREVIOUS_GAP_FIELD
        == sql_mod._SESSION_HISTORY_PREVIOUS_GAP_FIELD
    )


def test_rr006_shared_reasoning_and_tool_derivation_helpers() -> None:
    metadata = {"reasoning_content_present": True}
    message = {
        "content": [{"type": "text", "text": "done"}],
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "Read", "arguments": "{}"},
            }
        ],
    }
    reasoning = aawm_agent_identity._derive_session_history_reasoning_fields(
        metadata=metadata,
        message=message,
        resolved_model="claude-sonnet-4-6",
        provider="anthropic",
        reported_reasoning_tokens=4,
        provider_reported_reasoning_tokens=4,
    )
    assert reasoning["reasoning_present"] is True
    assert reasoning["reported_reasoning_tokens"] == 4
    assert reasoning["reasoning_tokens_source"] == "provider_reported"
    tools = aawm_agent_identity._derive_session_history_tool_fields(
        message=message,
        request_body={"tools": []},
        metadata={},
        output_payload=None,
    )
    assert tools["tool_call_count"] == 1
    assert tools["tool_names"] == ["Read"]
    assert isinstance(tools["tool_activity_summary"], dict)


def test_rr006_session_history_package_writer_reexport_parity() -> None:
    from litellm.integrations.aawm_session_history import writer
    from litellm.integrations.aawm_session_history import spool
    from litellm.integrations.aawm_session_history import retry
    from litellm.integrations.aawm_session_history import record
    from litellm.integrations.aawm_session_history import runtime
    from litellm.integrations import aawm_agent_identity as identity
    from litellm.integrations import aawm_session_history_sql as sql_shim
    from litellm.integrations.aawm_session_history import sql as sql_mod

    # Disjoint package modules own writer/spool/retry/record; identity re-exports.
    assert identity._enqueue_session_history_record is writer._enqueue_session_history_record
    assert identity._spool_session_history_records is spool._spool_session_history_records
    assert identity._spool_session_history_records is writer._spool_session_history_records
    assert identity._flush_session_history_batch_with_retry is (
        retry._flush_session_history_batch_with_retry
    )
    assert identity._flush_session_history_batch_with_retry is (
        writer._flush_session_history_batch_with_retry
    )
    assert identity._shutdown_session_history_worker is (
        writer._shutdown_session_history_worker
    )
    assert identity._aawm_session_history_queue is runtime._aawm_session_history_queue
    assert identity._aawm_session_history_queue is writer._aawm_session_history_queue
    assert identity._AAWM_SESSION_HISTORY_INSERT_SQL is sql_mod._AAWM_SESSION_HISTORY_INSERT_SQL
    assert sql_shim._AAWM_SESSION_HISTORY_TABLE_SQL is sql_mod._AAWM_SESSION_HISTORY_TABLE_SQL

    # Record builders are package-owned and re-exported on identity.
    assert identity._build_session_history_record is record._build_session_history_record
    assert identity._persist_session_history_records is (
        record._persist_session_history_records
    )
    assert identity._build_session_history_record.__module__.endswith(
        "aawm_session_history.record"
    )

    # Identity callback module must not still define the durable service.
    source = Path(identity.__file__).read_text(encoding="utf-8")
    assert "queue.Queue(maxsize=1024)" not in source
    assert "def _session_history_worker_main" not in source
    assert "def _enqueue_session_history_record" not in source
    assert "def _spool_session_history_records" not in source
    assert "def _flush_session_history_batch_with_retry" not in source
    assert "def _build_session_history_record(" not in source
    assert "def _persist_session_history_records(" not in source
    assert "from litellm.integrations.aawm_session_history.writer import" in source
    assert "aawm_session_history.record" in source or "session_history import record" in source




def test_rr006_record_apis_are_real_source_not_exec() -> None:
    """RR-006 gate: record APIs must be ordinary Python, not compile/exec text."""
    import inspect
    from litellm.integrations.aawm_session_history import record
    from litellm.integrations import aawm_agent_identity as identity

    record._ensure_installed()
    fn = identity._build_session_history_record
    assert fn is record._build_session_history_record
    assert fn.__module__.endswith("aawm_session_history.record")
    # Real file path in code object (not <aawm_session_history.record:...> synthetic).
    assert fn.__code__.co_filename.endswith("aawm_session_history/record.py")
    assert not fn.__code__.co_filename.startswith("<")
    src = inspect.getsource(fn)
    assert src.lstrip().startswith("def _build_session_history_record")
    # Package module must not keep the relocated-source-string mechanism.
    record_source = Path(record.__file__).read_text(encoding="utf-8")
    assert "_RECORD_SOURCES" not in record_source
    assert "compile(src" not in record_source
    assert "exec(code" not in record_source
    assert "FunctionType" in record_source
    # Class pollution from the old failure-event source blob must stay gone.
    assert "class AawmAgentIdentity" not in record_source


def test_rr006_record_api_monkeypatch_uses_identity_globals(monkeypatch) -> None:
    """Rebind keeps free-name lookup on aawm_agent_identity for tests/scripts."""
    from litellm.integrations.aawm_session_history import record
    from litellm.integrations import aawm_agent_identity as identity

    record._ensure_installed()
    fn = identity._build_session_history_record
    assert fn.__globals__ is identity.__dict__
    assert fn.__globals__ is not record.__dict__

    marker = object()

    def fake_safe_int(value):
        if value == "rr006-marker":
            return marker
        return identity.__dict__.get("_safe_int_original_probe", None)

    # Patch the shared helper on the identity host; record API globals must see it.
    monkeypatch.setattr(identity, "_safe_int", lambda value: 4242 if value == "rr006-marker" else None)
    assert fn.__globals__["_safe_int"]("rr006-marker") == 4242


def test_rr006_session_history_modules_are_independently_importable() -> None:
    from litellm.integrations.aawm_session_history import runtime
    from litellm.integrations.aawm_session_history import spool
    from litellm.integrations.aawm_session_history import retry
    from litellm.integrations.aawm_session_history import writer
    from litellm.integrations.aawm_session_history import record
    from litellm.integrations.aawm_session_history import sql

    # Each concern has a dedicated module surface that can be imported without
    # depending on wheel packaging and without redefining the same queue service.
    assert runtime._aawm_session_history_queue.maxsize == 1024
    assert callable(spool._spool_session_history_records)
    assert callable(retry._flush_session_history_batch_with_retry)
    assert callable(writer._enqueue_session_history_record)
    assert callable(record._build_session_history_record)
    assert isinstance(sql._AAWM_SESSION_HISTORY_INSERT_SQL, str)
    # Writer re-exports spool/retry for historical import paths.
    assert writer._spool_session_history_records is spool._spool_session_history_records
    assert writer._flush_session_history_batch_with_retry is (
        retry._flush_session_history_batch_with_retry
    )


def test_rr006_session_history_writer_state_bridge_uses_identity_monkeypatch(
    monkeypatch,
) -> None:
    from litellm.integrations.aawm_session_history import writer
    from litellm.integrations import aawm_agent_identity as identity

    class FakeQueue:
        def __init__(self) -> None:
            self.maxsize = 8
            self.items = []

        def put(self, item, timeout=None):
            self.items.append((item, timeout))

        def get_nowait(self):
            raise queue.Empty

    fake = FakeQueue()
    monkeypatch.setattr(identity, "_aawm_session_history_queue", fake)
    assert writer._state("_aawm_session_history_queue") is fake

    monkeypatch.setattr(
        identity,
        "_get_session_history_flush_interval_seconds",
        lambda: 0.5,
    )
    assert writer._call("_get_session_history_flush_interval_seconds") == 0.5

def test_rr006_rate_limit_context_identity_cached_across_sources(monkeypatch) -> None:
    calls = {"repository": 0, "tenant": 0}

    def fake_repo(kwargs, **_kwargs):
        calls["repository"] += 1
        return "repo-a"

    def fake_tenant(kwargs, **_kwargs):
        calls["tenant"] += 1
        return "tenant-a", "test"

    monkeypatch.setattr(
        aawm_agent_identity,
        "_extract_repository_identity_from_kwargs",
        fake_repo,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_extract_tenant_identity_from_kwargs",
        fake_tenant,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_merged_rate_limit_metadata",
        lambda kwargs: {},
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_resolve_rate_limit_model",
        lambda kwargs, result, metadata: "gpt-4o",
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_normalize_session_history_provider",
        lambda *args, **kwargs: "openai",
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_infer_rate_limit_client_family",
        lambda *args, **kwargs: "openai",
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_build_session_runtime_identity",
        lambda **kwargs: {
            "litellm_environment": "dev",
            "client_name": "codex",
            "client_version": "1",
            "client_user_agent": "ua",
        },
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_extract_rate_limit_account_hash",
        lambda *args, **kwargs: "acct",
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_extract_session_id",
        lambda kwargs: "sess",
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_extract_trace_id",
        lambda kwargs: "trace",
    )

    kwargs: dict[str, Any] = {"litellm_call_id": "call-1"}
    ctx1 = aawm_agent_identity._build_rate_limit_context(
        kwargs, None, "2026-07-17T00:00:00Z", "codex_token_count"
    )
    ctx2 = aawm_agent_identity._build_rate_limit_context(
        kwargs, None, "2026-07-17T00:00:00Z", "anthropic_response_headers"
    )
    assert ctx1["repository"] == "repo-a"
    assert ctx2["repository"] == "repo-a"
    assert calls["repository"] == 1
    assert calls["tenant"] == 1
    assert aawm_agent_identity._AAWM_RATE_LIMIT_CONTEXT_CACHE_KEY in kwargs


def test_rr006_chunks_literal_eval_rejects_oversized_payload() -> None:
    oversized = "Chunks=" + ("[" + ("1," * 5000) + "1]")
    assert (
        aawm_agent_identity._extract_responses_completed_payload_from_passthrough_fallback_text(
            oversized
        )
        is None
    )


def test_rr006_workspace_root_override_resolves_repo(monkeypatch) -> None:
    monkeypatch.setenv("AAWM_WORKSPACE_ROOT", "/data/workspaces")
    monkeypatch.setenv("AAWM_CODEX_MEMORY_ROOT", "/data/codex/memories")
    assert (
        aawm_agent_identity._normalize_repository_identity_from_absolute_path(
            "/data/workspaces/litellm"
        )
        == "litellm"
    )
    assert (
        aawm_agent_identity._normalize_repository_identity_from_absolute_path(
            "/data/codex/memories"
        )
        == aawm_agent_identity._CODEX_MEMORY_ROOT_REPOSITORY
    )
    repo, source = aawm_agent_identity._extract_repository_identity_from_text_with_source(
        "cwd: /data/workspaces/aawm-tap"
    )
    assert repo == "aawm-tap"
    assert source in {
        "text.cwd_assignment",
        "text.project_path",
        "text.workspace_directories",
        "text.cwd_tag",
        "text.environment_context.cwd",
        "text.agents_instructions",
    }


def test_rr006_rate_limit_snapshot_signature_include_reset_flag() -> None:
    observation = {
        "provider_resets_at": "2026-07-17T01:00:00Z",
        "used_percentage": 10.0,
        "remaining_requests": 5,
    }
    full = aawm_agent_identity._rate_limit_snapshot_signature(observation)
    body = aawm_agent_identity._rate_limit_snapshot_signature(
        observation, include_reset=False
    )
    assert full[1:] == body
    assert len(full) == len(body) + 1


def test_rr006_permission_check_does_not_special_case_unittest_mock() -> None:
    from unittest.mock import MagicMock

    source = Path(aawm_agent_identity.__file__).read_text(encoding="utf-8")
    assert 'startswith("unittest.mock")' not in source
    # Arbitrary mock without response-shaped fields must not be probed as content.
    assert (
        aawm_agent_identity._extract_claude_permission_check_decision_from_value(
            MagicMock()
        )
        is None
    )
    assert (
        aawm_agent_identity._extract_claude_permission_check_decision_from_value(
            {"content": "<block>yes"}
        )
        == "yes"
    )


def test_rr006_open_connection_skips_manual_application_name_init(monkeypatch) -> None:
    import asyncio
    from litellm.integrations.aawm_session_history import writer as sh_writer

    class _Conn:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    created: dict[str, Any] = {}
    init_calls = {"count": 0}

    class _AsyncPG:
        @staticmethod
        async def connect(**kwargs):
            created.update(kwargs)
            return _Conn()

    async def fake_init(conn):
        init_calls["count"] += 1

    # Writer helpers resolve through the identity host for test monkeypatching.
    monkeypatch.setattr(
        aawm_agent_identity,
        "_build_session_history_dsn",
        lambda: "postgresql://example/db",
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_get_session_history_command_timeout_seconds",
        lambda: 5,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_get_session_history_statement_cache_size",
        lambda: 0,
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_get_session_history_server_settings",
        lambda: {"application_name": "aawm-session-history"},
    )
    monkeypatch.setattr(
        aawm_agent_identity,
        "_initialize_session_history_connection",
        fake_init,
    )
    monkeypatch.setattr(sh_writer, "_initialize_session_history_connection", fake_init)

    class _Importlib:
        @staticmethod
        def import_module(name: str):
            assert name == "asyncpg"
            return _AsyncPG

    monkeypatch.setattr(sh_writer, "_writer_importlib", lambda: _Importlib())

    async def _run():
        return await sh_writer._open_aawm_session_history_connection()

    conn = asyncio.run(_run())
    assert created.get("server_settings") == {"application_name": "aawm-session-history"}
    assert init_calls["count"] == 0
    assert isinstance(conn, _Conn)



def test_rr006_package_root_exports_underscore_apis() -> None:
    """Package root must export private SQL/writer APIs via explicit __all__."""
    import litellm.integrations.aawm_session_history as package
    from litellm.integrations.aawm_session_history import writer
    from litellm.integrations.aawm_session_history import sql as sql_mod

    assert "_enqueue_session_history_record" in package.__all__
    assert "_AAWM_SESSION_HISTORY_INSERT_SQL" in package.__all__
    assert package._enqueue_session_history_record is writer._enqueue_session_history_record
    assert (
        package._AAWM_SESSION_HISTORY_INSERT_SQL
        is sql_mod._AAWM_SESSION_HISTORY_INSERT_SQL
    )
    assert package._aawm_session_history_queue is writer._aawm_session_history_queue
    assert package._flush_session_history_batch_with_retry is (
        writer._flush_session_history_batch_with_retry
    )
    # Star-import must include underscore names when __all__ lists them.
    namespace: dict[str, object] = {}
    exec("from litellm.integrations.aawm_session_history import *", namespace)
    assert "_enqueue_session_history_record" in namespace
    assert "_AAWM_SESSION_HISTORY_INSERT_SQL" in namespace


def test_rr006_repository_identity_skips_full_kwargs_deep_scan(monkeypatch) -> None:
    """RR-006 #18: do not deep-walk entire kwargs as a last-resort repository source."""
    visits: list[str] = []
    original = aawm_agent_identity._extract_repository_identity_from_value_with_source

    def tracking(value, *, source_prefix, _seen=None, _depth=0):
        visits.append(source_prefix)
        return original(value, source_prefix=source_prefix, _seen=_seen, _depth=_depth)

    monkeypatch.setattr(
        aawm_agent_identity,
        "_extract_repository_identity_from_value_with_source",
        tracking,
    )
    kwargs = {
        "litellm_params": {
            "metadata": {},
            "proxy_server_request": {
                "headers": {},
                "body": {
                    "metadata": {},
                    "messages": [{"role": "user", "content": "no repo here"}],
                },
            },
        },
        "standard_logging_object": {"metadata": {}},
        "passthrough_logging_payload": {
            "request_body": {
                "metadata": {},
                "instructions": "plain text without repository markers",
            }
        },
        "noise_blob": {"nested": [{"deep": "x" * 100}]},
    }
    repository, source = aawm_agent_identity._extract_repository_identity_from_kwargs_with_source(
        kwargs
    )
    assert repository is None
    assert source is None
    # Full-object last-resort fallbacks must not appear.
    assert not any(prefix == "kwargs" for prefix in visits)
    assert not any(prefix == "standard_logging_object" for prefix in visits)
    assert not any(prefix == "passthrough_logging_payload" for prefix in visits)
    assert not any(prefix.startswith("noise_blob") for prefix in visits)
    # Request body sources remain allowed for workspace-text inference.
    assert any(
        prefix.startswith("passthrough_logging_payload.request_body")
        or prefix.startswith("litellm_params.proxy_server_request.body")
        for prefix in visits
    )


def test_rr006_workspace_defaults_use_expanduser_not_hardcoded_home() -> None:
    """RR-006 #20: source defaults must be portable expanduser paths."""
    import os

    # Runtime values may expand to this developer's home; that is fine.
    assert aawm_agent_identity._AAWM_WORKSPACE_ROOT_DEFAULT == os.path.expanduser(
        "~/projects"
    )
    assert aawm_agent_identity._AAWM_CODEX_MEMORY_ROOT_DEFAULT == os.path.expanduser(
        "~/.codex/memories"
    )
    source = Path(aawm_agent_identity.__file__).read_text(encoding="utf-8")
    assert 'os.path.expanduser("~/projects")' in source
    assert 'os.path.expanduser("~/.codex/memories")' in source
    # Hardcoded absolute developer paths must not remain in source.
    assert '"/home/zepfu/projects"' not in source
    assert '"/home/zepfu/.codex/memories"' not in source
    assert "'/home/zepfu/projects'" not in source
    assert "'/home/zepfu/.codex/memories'" not in source


def test_rr006_ensure_session_history_schema_is_noop_without_ddl(monkeypatch) -> None:
    """RR-006 #40: hot-path schema ensure must not execute DDL statements."""
    import asyncio
    from litellm.integrations.aawm_session_history import writer
    from litellm.integrations.aawm_session_history import runtime

    monkeypatch.setattr(runtime, "_aawm_session_history_schema_ready", False, raising=False)
    monkeypatch.setattr(
        aawm_agent_identity, "_aawm_session_history_schema_ready", False, raising=False
    )

    class BoomConn:
        def __init__(self) -> None:
            self.executed: list[str] = []

        async def execute(self, sql, *args):
            self.executed.append(str(sql))
            raise AssertionError("schema ensure must not execute SQL")

    conn = BoomConn()

    async def _run() -> None:
        await writer._ensure_session_history_schema(conn)
        await writer._ensure_session_history_schema(conn)

    asyncio.run(_run())
    assert conn.executed == []
    assert runtime._state("_aawm_session_history_schema_ready") is True


def test_rr006_session_history_service_uses_threading_locks_only() -> None:
    """RR-006 #22: durable writer state is threading-locked, not asyncio-locked."""
    from litellm.integrations.aawm_session_history import runtime
    import threading

    assert isinstance(runtime._aawm_session_history_worker_lock, type(threading.Lock()))
    assert isinstance(runtime._aawm_session_history_pool_lock, type(threading.Lock()))
    assert isinstance(runtime._aawm_session_history_schema_lock, type(threading.Lock()))
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "threading-only" in source
    assert "asyncio.Lock" not in source


def test_rr006_wheel_packaging_force_includes_session_history_package() -> None:
    """RR-006 #3: callback wheel packaging must ship session_history modules."""
    pyproject = Path(".wheel-build/pyproject.toml").read_text(encoding="utf-8")
    required = [
        'aawm_litellm_callbacks/agent_identity.py',
        "litellm/integrations/aawm_session_history/__init__.py",
        "litellm/integrations/aawm_session_history/runtime.py",
        "litellm/integrations/aawm_session_history/writer.py",
        "litellm/integrations/aawm_session_history/spool.py",
        "litellm/integrations/aawm_session_history/retry.py",
        "litellm/integrations/aawm_session_history/record.py",
        "litellm/integrations/aawm_session_history/sql.py",
        "litellm/integrations/aawm_session_history/identity_selection.py",
        "litellm/integrations/aawm_session_history_sql.py",
    ]
    for marker in required:
        assert marker in pyproject, marker
