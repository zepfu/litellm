from __future__ import annotations

import json
from datetime import datetime, timezone

import scripts.backfill_rate_limit_observations as quota_backfill


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
