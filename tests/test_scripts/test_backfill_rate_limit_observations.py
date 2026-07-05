from __future__ import annotations

import json
from datetime import datetime, timezone

import scripts.backfill_rate_limit_observations as quota_backfill


def test_should_prefer_direct_dsn_for_rate_limit_backfill(monkeypatch) -> None:
    monkeypatch.setattr(
        quota_backfill,
        "_get_first_secret",
        lambda names: (
            "postgresql://aawm:aawm_dev@postgres18:5432/aawm_tristore"
            if "AAWM_DIRECT_DATABASE_URL" in names
            else None
        ),
    )
    monkeypatch.setattr(
        quota_backfill,
        "_build_aawm_dsn",
        lambda: "postgresql://aawm:aawm_dev@pgbouncer:6432/aawm_tristore",
    )

    assert quota_backfill._build_aawm_admin_dsn() == (
        "postgresql://aawm:aawm_dev@postgres18:5432/aawm_tristore"
    )


def test_should_format_clickhouse_datetime_without_timezone_suffix() -> None:
    assert (
        quota_backfill._format_clickhouse_datetime(
            datetime(2026, 5, 5, 15, 30, tzinfo=timezone.utc)
        )
        == "2026-05-05 15:30:00"
    )


def test_should_extract_codex_rate_limits_from_structured_clickhouse_output() -> None:
    row = {
        "observation_id": "obs-codex",
        "observation_trace_id": "trace-codex",
        "observation_start_time": "2026-05-05T15:00:00Z",
        "observation_end_time": "2026-05-05T15:00:01Z",
        "observation_name": "litellm-pass_through_endpoint",
        "observation_metadata": {
            "client_name": "codex",
            "passthrough_route_family": "codex_responses",
        },
        "observation_input": None,
        "observation_output": json.dumps(
            {
                "rate_limits": {
                    "limit_id": "codex_bengalfox",
                    "limit_name": "GPT-5.3-Codex-Spark",
                    "primary": {
                        "used_percent": 80.0,
                        "window_minutes": 300,
                        "resets_at": 1778000000,
                    },
                    "secondary": {
                        "used_percent": 100.0,
                        "window_minutes": 10080,
                        "resets_at": 1778018910,
                    },
                }
            }
        ),
        "observation_model": "gpt-5.3-codex-spark",
        "observation_environment": "dev",
    }

    record = quota_backfill.build_record_from_clickhouse_row(row)

    assert record is not None
    observations = record["rate_limit_observations"]
    assert len(observations) == 2
    assert {observation["limit_scope"] for observation in observations} == {
        "primary",
        "secondary",
    }
    assert all(
        observation["evidence"]["historical_backfill"] is True
        for observation in observations
    )
    assert observations[0]["evidence"]["backfill_source"] == "langfuse_clickhouse"
    assert observations[0]["observed_at"] == datetime(
        2026,
        5,
        5,
        15,
        0,
        1,
        tzinfo=timezone.utc,
    )


def test_should_dedupe_backfill_observations_independent_of_call_id() -> None:
    base = {
        "source": "codex_response_headers",
        "provider": "openai",
        "model": "gpt-5.5",
        "limit_id": "codex",
        "limit_scope": "primary",
        "observed_at": datetime(2026, 5, 5, 15, 0, 1, tzinfo=timezone.utc),
        "provider_resets_at": datetime(2026, 5, 5, 20, 0, tzinfo=timezone.utc),
        "used_percentage": 42.0,
        "trace_id": "trace-codex",
    }

    live_observation = dict(base, litellm_call_id="live-call")
    backfill_observation = dict(base, litellm_call_id="time-15-00-00_backfill-call")

    assert quota_backfill._observation_signature(live_observation) == (
        quota_backfill._observation_signature(backfill_observation)
    )


def test_should_dedupe_backfill_observations_at_millisecond_precision() -> None:
    base = {
        "source": "codex_response_headers",
        "provider": "openai",
        "model": "gpt-5.5",
        "limit_id": "codex",
        "limit_scope": "primary",
        "provider_resets_at": datetime(
            2026, 5, 5, 20, 0, 0, 123456, tzinfo=timezone.utc
        ),
        "used_percentage": 42.0,
        "trace_id": "trace-codex",
    }

    live_observation = dict(
        base,
        observed_at=datetime(2026, 5, 5, 15, 0, 1, 641123, tzinfo=timezone.utc),
        litellm_call_id="live-call",
    )
    backfill_observation = dict(
        base,
        observed_at=datetime(2026, 5, 5, 15, 0, 1, 641999, tzinfo=timezone.utc),
        litellm_call_id="time-15-00-00_backfill-call",
    )

    assert quota_backfill._observation_signature(live_observation) == (
        quota_backfill._observation_signature(backfill_observation)
    )


def test_should_ignore_assistant_prose_keyword_matches() -> None:
    row = {
        "observation_id": "obs-prose",
        "observation_trace_id": "trace-prose",
        "observation_start_time": "2026-05-05T15:00:00Z",
        "observation_end_time": "2026-05-05T15:00:01Z",
        "observation_name": "litellm-pass_through_endpoint",
        "observation_metadata": {},
        "observation_input": None,
        "observation_output": json.dumps(
            {
                "content": (
                    "The response mentioned rate_limits, resets_at, "
                    "usage_limit_reached, and retrieveUserQuota in prose."
                )
            }
        ),
        "observation_model": "gpt-5.5",
        "observation_environment": "dev",
    }

    assert quota_backfill.build_record_from_clickhouse_row(row) is None


def test_should_extract_google_quota_from_structured_metadata() -> None:
    row = {
        "observation_id": "obs-google",
        "observation_trace_id": "trace-google",
        "observation_start_time": "2026-05-05T15:00:00Z",
        "observation_end_time": "2026-05-05T15:00:01Z",
        "observation_name": "native_gemini_passthrough",
        "observation_metadata": {
            "custom_llm_provider": "gemini",
            "passthrough_route_family": "google_code_assist",
            "google_retrieve_user_quota": {
                "remainingRequests": 1490,
                "usedRequests": 10,
                "totalRequests": 1500,
                "quotaPeriod": "daily",
            },
        },
        "observation_input": None,
        "observation_output": None,
        "observation_model": "gemini-2.5-pro",
        "observation_environment": "dev",
    }

    record = quota_backfill.build_record_from_clickhouse_row(row)

    assert record is not None
    [observation] = record["rate_limit_observations"]
    assert observation["provider"] == "gemini"
    assert observation["client_family"] == "google_code_assist"
    assert observation["remaining_requests"] == 1490
    assert observation["used_requests"] == 10
    assert observation["total_requests"] == 1500
    assert observation["used_percentage"] == 10 / 1500 * 100


def test_should_extract_anthropic_quota_from_structured_metadata() -> None:
    row = {
        "observation_id": "obs-anthropic",
        "observation_trace_id": "trace-anthropic",
        "observation_start_time": "2026-05-05T15:00:00Z",
        "observation_end_time": "2026-05-05T15:00:01Z",
        "observation_name": "litellm-pass_through_endpoint",
        "observation_metadata": {
            "client_name": "claude-cli",
            "passthrough_route_family": "anthropic_messages",
            "anthropic_response_headers": {
                "source": "anthropic_response_headers",
                "anthropic-ratelimit-unified-5h-reset": "1778034000",
                "anthropic-ratelimit-unified-5h-status": "allowed",
                "anthropic-ratelimit-unified-5h-utilization": "0.42",
            },
        },
        "observation_input": None,
        "observation_output": None,
        "observation_model": "claude-sonnet-4-6",
        "observation_environment": "dev",
    }

    record = quota_backfill.build_record_from_clickhouse_row(row)

    assert record is not None
    [observation] = record["rate_limit_observations"]
    assert observation["provider"] == "anthropic"
    assert observation["client_family"] == "claude"
    assert observation["limit_scope"] == "5h"
    assert observation["used_percentage"] == 42.0




def test_should_extract_anthropic_7d_oi_from_structured_metadata() -> None:
    row = {
        "observation_id": "obs-anthropic-7d-oi",
        "observation_trace_id": "trace-anthropic-7d-oi",
        "observation_start_time": "2026-07-02T14:30:00Z",
        "observation_end_time": "2026-07-02T14:30:01Z",
        "observation_name": "litellm-pass_through_endpoint",
        "observation_metadata": {
            "client_name": "claude-cli",
            "passthrough_route_family": "anthropic_messages",
            "anthropic_response_headers": {
                "source": "anthropic_response_headers",
                "anthropic-ratelimit-unified-7d_oi-reset": "1783036800",
                "anthropic-ratelimit-unified-7d_oi-status": "allowed_warning",
                "anthropic-ratelimit-unified-7d_oi-utilization": "0.88",
                "anthropic-ratelimit-unified-representative-claim": "seven_day_overage_included",
                "anthropic-ratelimit-unified-overage-status": "approaching_limit",
            },
        },
        "observation_input": None,
        "observation_output": None,
        "observation_model": "claude-fable-5",
        "observation_environment": "dev",
    }

    record = quota_backfill.build_record_from_clickhouse_row(row)

    assert record is not None
    oi_rows = [
        o
        for o in record["rate_limit_observations"]
        if o.get("limit_scope") == "7d_oi"
    ]
    assert len(oi_rows) == 1
    observation = oi_rows[0]
    assert observation["provider"] == "anthropic"
    assert observation["limit_id"] == "anthropic_unified_7d_oi"
    assert observation["quota_period"] == "seven_day"
    assert observation["used_percentage"] == 88.0
    assert observation["raw_provider_fields"][
        "anthropic-ratelimit-unified-representative-claim"
    ] == "seven_day_overage_included"


def test_should_skip_anthropic_7d_oi_when_reset_header_stale() -> None:
    row = {
        "observation_id": "obs-anthropic-7d-oi-stale",
        "observation_trace_id": "trace-anthropic-7d-oi-stale",
        "observation_start_time": "2026-07-03T12:00:00Z",
        "observation_end_time": "2026-07-03T12:00:01Z",
        "observation_name": "litellm-pass_through_endpoint",
        "observation_metadata": {
            "client_name": "claude-cli",
            "anthropic_response_headers": {
                "source": "anthropic_response_headers",
                "anthropic-ratelimit-unified-7d_oi-reset": "2026-05-14T02:40:00Z",
                "anthropic-ratelimit-unified-7d_oi-status": "allowed",
                "anthropic-ratelimit-unified-7d_oi-utilization": "0.42",
            },
        },
        "observation_input": None,
        "observation_output": None,
        "observation_model": "claude-fable-5",
        "observation_environment": "dev",
    }

    record = quota_backfill.build_record_from_clickhouse_row(row)

    if record is None:
        assert True
    else:
        assert not any(
            o.get("limit_scope") == "7d_oi"
            for o in record.get("rate_limit_observations", [])
        )


def test_should_extract_xai_oauth_quota_from_structured_metadata() -> None:
    row = {
        "observation_id": "obs-xai-oauth",
        "observation_trace_id": "trace-xai-oauth",
        "observation_start_time": "2026-06-02T06:00:00Z",
        "observation_end_time": "2026-06-02T06:00:01Z",
        "observation_name": "litellm-pass_through_endpoint",
        "observation_metadata": {
            "credential_family": "xai_oauth",
            "passthrough_route_family": "xai_oauth_api",
            "xai_oauth_managed": True,
            "xai_oauth_public_model": "oa_xai/grok-4.3",
            "xai_oauth_upstream_model": "xai/grok-4.3",
            "xai_oauth_response_headers": {
                "source": "xai_oauth_response_headers",
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "97",
                "x-ratelimit-limit-tokens": "15000000",
                "x-ratelimit-remaining-tokens": "14925000",
                "config": {
                    "billingPeriodEnd": "2026-07-01T00:00:00+00:00",
                },
            },
        },
        "observation_input": None,
        "observation_output": None,
        "observation_model": "xai/grok-4.3",
        "observation_environment": "dev",
    }

    record = quota_backfill.build_record_from_clickhouse_row(row)

    assert record is not None
    observations = record["rate_limit_observations"]
    assert len(observations) == 2
    by_scope = {observation["limit_scope"]: observation for observation in observations}
    assert by_scope["requests"]["provider"] == "xai"
    assert by_scope["requests"]["client_family"] == "xai_oauth"
    assert by_scope["requests"]["model"] == "oa_xai/grok-4.3"
    assert by_scope["requests"]["remaining_pct"] == 97.0
    assert by_scope["requests"]["provider_resets_at"] == datetime(
        2026, 7, 1, tzinfo=timezone.utc
    )
    assert by_scope["tokens"]["quota_type"] == "tokens"
    assert by_scope["tokens"]["remaining_pct"] == 99.5
    assert by_scope["tokens"]["provider_resets_at"] == datetime(
        2026, 7, 1, tzinfo=timezone.utc
    )
