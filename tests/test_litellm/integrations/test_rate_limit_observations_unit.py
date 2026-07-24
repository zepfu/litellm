"""Wave A3A (rate-limit typed extraction) parity / golden tests.

`litellm/integrations/aawm_agent_identity/__init__.py` CURRENTLY holds the
rate-limit / provider-error extraction band inline. Wave A3A's engineer will
extract it into `rate_limit_base.py`, `rate_limit_providers.py`, and
`provider_errors.py` behind a typed `ObservationExtractor` Protocol and a
frozen `RateLimitObservation` dataclass, leaving façade rebinds in the package
`__init__` so every existing importer keeps working.

These tests are written BEFORE the move and PIN current behavior. They are
parity/golden tests for an extraction landing: they must PASS on develop today
(capturing the exact observation each extractor produces and the exact DB
payload tuple `_build_rate_limit_observation_db_payload` derives from it) and
must STILL PASS after the engineer relocates the code, because the
record-dict input/output contract and the DB payload tuple shapes are UNCHANGED
this wave (see `.analysis/plan-godmodule-decomposition-r3-remediation-2026-07-23.md`
### Wave A3 -> Test Spec).

Invariants pinned here:
  * Each provider extractor maps a realistic captured-shape candidate (header
    map / billing payload / quota payload / free-model 429) to an exact
    observation dict -- every field, including the derived `limit_key`,
    `quota_period`, `inferred_window_start_at`, and the storage-facing
    `remaining_pct` / `quota_*` values.
  * `_build_rate_limit_observation_db_payload(record)` maps each observation to
    a byte-identical 22-tuple. This is the storage contract the extraction must
    not change.

Symbols are imported through the public identity namespace
(`litellm.integrations.aawm_agent_identity`), the same surface `scripts/` and
the existing identity tests use, so the tests keep working once the engineer
adds the A3A façades.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from litellm.integrations.aawm_agent_identity import (
    _build_rate_limit_observation_db_payload,
    _extract_anthropic_header_rate_limit_observations,
    _extract_codex_header_rate_limit_observations,
    _extract_google_quota_observations,
    _extract_grok_billing_observations,
    _extract_openrouter_free_error_observations,
    _extract_xai_oauth_header_rate_limit_observations,
)

# Fixed observation anchor. Every reset timestamp in the fixtures is in the
# future relative to this instant, so no observation is dropped as "stale"
# (the extractor discards resets older than the 15-minute stale tolerance).
OBSERVED_AT = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

# Environment variables that `_build_session_runtime_identity` reads (with
# allow_runtime=True) to populate the observation `environment` field. Cleared
# per-test so the golden `environment: None` is hermetic regardless of the
# runner's ambient configuration.
_ENVIRONMENT_ENV_VARS = (
    "AAWM_LITELLM_ENVIRONMENT",
    "LITELLM_INSTANCE_ENVIRONMENT",
    "LITELLM_ENVIRONMENT",
    "LITELLM_ENV",
    "LITELLM_LANGFUSE_TRACE_ENVIRONMENT",
    "LANGFUSE_TRACING_ENVIRONMENT",
    "AAWM_ENVIRONMENT",
)


@pytest.fixture(autouse=True)
def _clear_environment_env_vars(monkeypatch):
    """Make the identity-context `environment` field deterministically None."""
    for env_var in _ENVIRONMENT_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


# ---------------------------------------------------------------------------
# Deterministic captured-shape inputs (one per provider extractor).
# ---------------------------------------------------------------------------


def _base_kwargs(model, custom_llm_provider, metadata=None):
    """Minimal deterministic kwargs: model + provider + fixed call id.

    Rate-limit header maps are injected via ``litellm_params.metadata`` keys
    that ``_rate_limit_candidate_roots`` surfaces as candidate dicts. No
    standard_logging_object / session / trace / tenant identifiers are set, so
    every identity-context field resolves to a stable ``None``.
    """
    return {
        "model": model,
        "custom_llm_provider": custom_llm_provider,
        "litellm_call_id": "call-fixed-1",
        "litellm_params": {"metadata": dict(metadata or {})},
    }


class _OpenRouterFreeResponse:
    """A 429 upstream error body that classifies as ``rate_limited``."""

    status_code = 429
    text = '{"error": {"message": "Rate limit exceeded", "code": 429}}'


def _codex_inputs():
    metadata = {
        "codex_response_headers": {
            "x-codex-active-limit": "plus",
            "x-codex-primary-reset-at": "2026-07-01T17:00:00Z",
            "x-codex-primary-used-percent": "42.5",
            "x-codex-primary-window-minutes": "300",
        }
    }
    return _base_kwargs("gpt-5", "openai", metadata), {}


def _anthropic_inputs():
    metadata = {
        "anthropic_response_headers": {
            "anthropic-ratelimit-requests-limit": "1000",
            "anthropic-ratelimit-requests-remaining": "950",
            "anthropic-ratelimit-requests-reset": "2026-07-01T12:01:00Z",
        }
    }
    return _base_kwargs("claude-3-5-sonnet", "anthropic", metadata), {}


def _xai_oauth_inputs():
    metadata = {
        "credential_family": "xai_oauth",
        "xai_oauth_public_model": "grok-4",
        "xai_oauth_response_headers": {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "60",
            "x-ratelimit-reset-requests": "2026-07-01T13:00:00Z",
        },
    }
    return _base_kwargs("grok-4", "xai", metadata), {}


def _grok_billing_inputs():
    metadata = {
        "grok_cli_chat_proxy": True,
        "google_retrieve_user_quota": {
            "config": {
                "monthlyLimit": {"val": 1000},
                "used": {"val": 250},
                "billingPeriodStart": "2026-07-01T00:00:00Z",
                "billingPeriodEnd": "2026-08-01T00:00:00Z",
            }
        },
    }
    return _base_kwargs("grok-build", "xai", metadata), {}


def _openrouter_free_inputs():
    return (
        _base_kwargs("google/gemma-4-31b-it:free", "openrouter", {}),
        _OpenRouterFreeResponse(),
    )


def _google_quota_inputs():
    metadata = {
        "google_retrieve_user_quota": {
            "modelId": "gemini-2.5-pro",
            "remainingRequests": 80,
            "totalRequests": 100,
            "resetsAt": "2026-07-02T00:00:00Z",
        }
    }
    return _base_kwargs("gemini-2.5-pro", "gemini", metadata), {}


# ---------------------------------------------------------------------------
# Immutable pre-move golden observations + DB payload tuples.
# Captured by running the extractors above against the inputs above on develop
# (integration branch eff22ab9d2). Do not hand-edit: regenerate if the
# extraction contract intentionally changes.
# ---------------------------------------------------------------------------
CODEX_OBSERVATION = {
    "observed_at": datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    "provider": "openai",
    "client_family": "codex",
    "account_hash": None,
    "environment": None,
    "tenant_id": None,
    "repository": None,
    "session_id": None,
    "trace_id": None,
    "litellm_call_id": "call-fixed-1",
    "route_family": None,
    "request_model": None,
    "response_model": None,
    "model": "gpt-5",
    "model_family": "openai",
    "model_tier": None,
    "client_name": None,
    "client_version": None,
    "client_user_agent": None,
    "metadata": {},
    "source": "codex_response_headers",
    "limit_id": "codex",
    "limit_name": "Codex plus",
    "limit_scope": "primary",
    "window_minutes": 300,
    "provider_resets_at": datetime(2026, 7, 1, 17, 0, 0, tzinfo=timezone.utc),
    "used_percentage": 42.5,
    "reset_hint_seconds": None,
    "exhausted": False,
    "raw_provider_fields": {
        "x-codex-primary-reset-at": "2026-07-01T17:00:00Z",
        "x-codex-primary-reset-after-seconds": None,
        "x-codex-primary-over-secondary-limit-percent": None,
        "x-codex-primary-used-percent": "42.5",
        "x-codex-primary-window-minutes": "300",
        "x-codex-active-limit": "plus",
        "x-codex-credits-unlimited": None,
    },
    "evidence": {
        "signals": [
            "codex_response_rate_limit_headers",
        ],
        "provider_fields": [
            "x-codex-primary-reset-at",
            "x-codex-primary-reset-after-seconds",
            "x-codex-primary-used-percent",
            "x-codex-primary-window-minutes",
            "x-codex-primary-over-secondary-limit-percent",
        ],
    },
    "quota_period": "five_hour",
    "inferred_window_start_at": datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    "remaining_requests": None,
    "used_requests": None,
    "total_requests": None,
    "limit_key": "openai:codex:unknown_account:codex:primary:300",
    "status": "observed",
}

CODEX_DB_PAYLOAD = (
    datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    "codex",
    None,
    None,
    "openai",
    "gpt-5",
    "codex:primary",
    "five_hour",
    "tokens",
    datetime(2026, 7, 1, 17, 0, 0, tzinfo=timezone.utc),
    57.5,
    None,
    None,
    None,
    None,
    None,
    '{"x-codex-primary-reset-at": "2026-07-01T17:00:00Z", "x-codex-primary-reset-after-seconds": null, "x-codex-primary-over-secondary-limit-percent": null, "x-codex-primary-used-percent": "42.5", "x-codex-primary-window-minutes": "300", "x-codex-active-limit": "plus", "x-codex-credits-unlimited": null}',
    '{"signals": ["codex_response_rate_limit_headers"], "provider_fields": ["x-codex-primary-reset-at", "x-codex-primary-reset-after-seconds", "x-codex-primary-used-percent", "x-codex-primary-window-minutes", "x-codex-primary-over-secondary-limit-percent"]}',
    "codex_response_headers",
    None,
    None,
    "call-fixed-1",
)

ANTHROPIC_OBSERVATION = {
    "observed_at": datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    "provider": "anthropic",
    "client_family": "anthropic",
    "account_hash": None,
    "environment": None,
    "tenant_id": None,
    "repository": None,
    "session_id": None,
    "trace_id": None,
    "litellm_call_id": "call-fixed-1",
    "route_family": None,
    "request_model": None,
    "response_model": None,
    "model": "claude-3-5-sonnet",
    "model_family": "claude",
    "model_tier": "sonnet",
    "client_name": None,
    "client_version": None,
    "client_user_agent": None,
    "metadata": {},
    "source": "anthropic_response_headers",
    "limit_id": "anthropic_requests",
    "limit_name": "Anthropic requests rate limit",
    "limit_scope": "requests",
    "provider_resets_at": datetime(2026, 7, 1, 12, 1, 0, tzinfo=timezone.utc),
    "used_percentage": 5.0,
    "remaining_requests": 950,
    "used_requests": 50,
    "total_requests": 1000,
    "raw_provider_fields": {
        "anthropic-ratelimit-requests-limit": "1000",
        "anthropic-ratelimit-requests-remaining": "950",
        "anthropic-ratelimit-requests-reset": "2026-07-01T12:01:00Z",
    },
    "evidence": {
        "signals": [
            "anthropic_response_rate_limit_headers",
        ],
        "provider_fields": [
            "anthropic-ratelimit-requests-limit",
            "anthropic-ratelimit-requests-remaining",
            "anthropic-ratelimit-requests-reset",
        ],
    },
    "window_minutes": None,
    "quota_period": None,
    "inferred_window_start_at": None,
    "reset_hint_seconds": None,
    "limit_key": "anthropic:anthropic:unknown_account:anthropic_requests:requests:unknown_window",
    "exhausted": False,
    "status": "observed",
}

ANTHROPIC_DB_PAYLOAD = (
    datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    "anthropic",
    None,
    None,
    "anthropic",
    "claude-3-5-sonnet",
    "anthropic_requests:requests",
    None,
    "requests",
    datetime(2026, 7, 1, 12, 1, 0, tzinfo=timezone.utc),
    95.0,
    1000.0,
    50.0,
    950.0,
    None,
    None,
    '{"anthropic-ratelimit-requests-limit": "1000", "anthropic-ratelimit-requests-remaining": "950", "anthropic-ratelimit-requests-reset": "2026-07-01T12:01:00Z"}',
    '{"signals": ["anthropic_response_rate_limit_headers"], "provider_fields": ["anthropic-ratelimit-requests-limit", "anthropic-ratelimit-requests-remaining", "anthropic-ratelimit-requests-reset"]}',
    "anthropic_response_headers",
    None,
    None,
    "call-fixed-1",
)

XAI_OAUTH_OBSERVATION = {
    "observed_at": datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    "provider": "xai",
    "client_family": "xai_oauth",
    "account_hash": None,
    "environment": None,
    "tenant_id": None,
    "repository": None,
    "session_id": None,
    "trace_id": None,
    "litellm_call_id": "call-fixed-1",
    "route_family": None,
    "request_model": None,
    "response_model": None,
    "model": "grok-4",
    "model_family": "grok",
    "model_tier": None,
    "client_name": None,
    "client_version": None,
    "client_user_agent": None,
    "metadata": {
        "credential_family": "xai_oauth",
        "xai_oauth_public_model": "grok-4",
    },
    "source": "xai_oauth_response_headers",
    "limit_id": "xai_oauth_requests",
    "limit_name": "xAI OAuth requests rate limit",
    "limit_scope": "requests",
    "quota_period": None,
    "quota_type": "requests",
    "provider_resets_at": datetime(2026, 7, 1, 13, 0, 0, tzinfo=timezone.utc),
    "remaining_pct": 60.0,
    "quota_limit": 100.0,
    "quota_used": 40.0,
    "quota_remaining": 60.0,
    "billing_period_end_at": None,
    "used_percentage": 40.0,
    "remaining_requests": 60,
    "used_requests": 40,
    "total_requests": 100,
    "status": "observed",
    "exhausted": False,
    "exhaustion_kind": None,
    "reset_hint_seconds": None,
    "raw_provider_fields": {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "60",
        "reset": "2026-07-01T13:00:00Z",
        "retry-after": None,
        "billingPeriodEnd": None,
        "quota_unit": "xai_oauth_requests",
        "quota_unit_interpretation": "requests",
    },
    "evidence": {
        "signals": [
            "xai_oauth_response_rate_limit_headers",
        ],
        "provider_fields": [
            "x-ratelimit-limit-requests",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-request",
            "x-ratelimit-reset",
            "retry-after",
        ],
        "reset_absent": False,
        "reset_header_absent": False,
        "reset_source": "response_header",
    },
    "window_minutes": None,
    "inferred_window_start_at": None,
    "limit_key": "xai:xai_oauth:unknown_account:xai_oauth_requests:requests:unknown_window",
}

XAI_OAUTH_DB_PAYLOAD = (
    datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    "xai_oauth",
    None,
    None,
    "xai",
    "grok-4",
    "xai_oauth_requests:requests",
    None,
    "requests",
    datetime(2026, 7, 1, 13, 0, 0, tzinfo=timezone.utc),
    60.0,
    100.0,
    40.0,
    60.0,
    None,
    None,
    '{"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "60", "reset": "2026-07-01T13:00:00Z", "retry-after": null, "billingPeriodEnd": null, "quota_unit": "xai_oauth_requests", "quota_unit_interpretation": "requests"}',
    '{"signals": ["xai_oauth_response_rate_limit_headers"], "provider_fields": ["x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests", "x-ratelimit-reset-request", "x-ratelimit-reset", "retry-after"], "reset_absent": false, "reset_header_absent": false, "reset_source": "response_header"}',
    "xai_oauth_response_headers",
    None,
    None,
    "call-fixed-1",
)

GROK_BILLING_OBSERVATION = {
    "observed_at": datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    "provider": "xai",
    "client_family": "grok-build",
    "account_hash": None,
    "environment": None,
    "tenant_id": None,
    "repository": None,
    "session_id": None,
    "trace_id": None,
    "litellm_call_id": "call-fixed-1",
    "route_family": None,
    "request_model": None,
    "response_model": None,
    "model": "grok-build",
    "model_family": "grok",
    "model_tier": None,
    "client_name": None,
    "client_version": None,
    "client_user_agent": None,
    "metadata": {},
    "source": "grok_billing",
    "limit_id": "xai_grok_build_monthly_requests",
    "limit_name": "Grok Build monthly requests",
    "limit_scope": "requests",
    "quota_period": "monthly",
    "quota_type": "requests",
    "provider_resets_at": datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc),
    "remaining_pct": 75.0,
    "quota_limit": 1000.0,
    "quota_used": 250.0,
    "quota_remaining": 750.0,
    "billing_period_start_at": datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
    "billing_period_end_at": datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc),
    "used_percentage": 25.0,
    "raw_provider_fields": {
        "monthlyLimit": {
            "val": 1000,
        },
        "used": {
            "val": 250,
        },
        "onDemandCap": None,
        "billingPeriodStart": "2026-07-01T00:00:00Z",
        "billingPeriodEnd": "2026-08-01T00:00:00Z",
        "quota_unit": "grok_billing_used",
        "quota_unit_interpretation": "requests",
    },
    "evidence": {
        "signals": [
            "grok_billing_payload",
            "grok_billing_monthly_counter",
        ],
        "provider_fields": [
            "config.monthlyLimit.val",
            "config.used.val",
            "config.billingPeriodEnd",
        ],
        "rounding": "whole_remaining_percentage",
        "unit_note": "Grok billing does not label used.val; observed tool traffic behaves request-like.",
    },
    "window_minutes": None,
    "inferred_window_start_at": None,
    "remaining_requests": None,
    "used_requests": None,
    "total_requests": None,
    "reset_hint_seconds": None,
    "limit_key": "xai:grok-build:unknown_account:xai_grok_build_monthly_requests:requests:unknown_window",
    "exhausted": False,
    "status": "observed",
}

GROK_BILLING_DB_PAYLOAD = (
    datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    "grok-build",
    None,
    None,
    "xai",
    "grok-build",
    "xai_grok_build_monthly_requests:requests",
    "monthly",
    "requests",
    datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc),
    75.0,
    1000.0,
    250.0,
    750.0,
    datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc),
    '{"monthlyLimit": {"val": 1000}, "used": {"val": 250}, "onDemandCap": null, "billingPeriodStart": "2026-07-01T00:00:00Z", "billingPeriodEnd": "2026-08-01T00:00:00Z", "quota_unit": "grok_billing_used", "quota_unit_interpretation": "requests"}',
    '{"signals": ["grok_billing_payload", "grok_billing_monthly_counter"], "provider_fields": ["config.monthlyLimit.val", "config.used.val", "config.billingPeriodEnd"], "rounding": "whole_remaining_percentage", "unit_note": "Grok billing does not label used.val; observed tool traffic behaves request-like."}',
    "grok_billing",
    None,
    None,
    "call-fixed-1",
)

OPENROUTER_FREE_OBSERVATION = {
    "observed_at": datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    "provider": "openrouter",
    "client_family": "openrouter",
    "account_hash": "7a75195d187a",
    "environment": None,
    "tenant_id": None,
    "repository": None,
    "session_id": None,
    "trace_id": None,
    "litellm_call_id": "call-fixed-1",
    "route_family": None,
    "request_model": None,
    "response_model": None,
    "model": None,
    "model_family": "openrouter",
    "model_tier": "free",
    "client_name": None,
    "client_version": None,
    "client_user_agent": None,
    "metadata": {},
    "source": "openrouter_free_daily_local_meter",
    "limit_id": "openrouter_free_daily_requests",
    "limit_name": "OpenRouter free daily requests",
    "limit_scope": "requests",
    "window_minutes": 1440,
    "quota_period": "daily",
    "quota_type": "requests",
    "provider_resets_at": datetime(2026, 7, 2, 0, 0, 0, tzinfo=timezone.utc),
    "remaining_pct": 0.0,
    "used_percentage": 100.0,
    "remaining_requests": 0,
    "used_requests": 1000,
    "total_requests": 1000,
    "status": "quota_exhausted",
    "exhausted": True,
    "exhaustion_kind": "request_quota",
    "reset_hint_seconds": None,
    "raw_provider_fields": {
        "dailyLimit": 1000,
        "usedRequests": 1000,
        "remainingRequests": 0,
        "windowStart": "2026-07-01T00:00:00Z",
        "windowEnd": "2026-07-02T00:00:00Z",
        "reset_anchor": "utc_midnight",
        "model_scope": "openrouter_:free_shared_pool",
        "meter_source": "local_session_history",
    },
    "evidence": {
        "signals": [
            "openrouter_free_model_rate_limit_error",
        ],
        "provider_fields": [],
        "scope_note": "OpenRouter documents free-model quota as account-level; provider does not expose current free request usage.",
    },
    "inferred_window_start_at": datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
    "limit_key": "openrouter:openrouter:7a75195d187a:openrouter_free_daily_requests:requests:1440",
}

OPENROUTER_FREE_DB_PAYLOAD = (
    datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    "openrouter",
    None,
    "7a75195d187a",
    "openrouter",
    None,
    "openrouter_free_daily_requests:requests",
    "daily",
    "requests",
    datetime(2026, 7, 2, 0, 0, 0, tzinfo=timezone.utc),
    0.0,
    1000.0,
    1000.0,
    0.0,
    None,
    None,
    '{"dailyLimit": 1000, "usedRequests": 1000, "remainingRequests": 0, "windowStart": "2026-07-01T00:00:00Z", "windowEnd": "2026-07-02T00:00:00Z", "reset_anchor": "utc_midnight", "model_scope": "openrouter_:free_shared_pool", "meter_source": "local_session_history"}',
    '{"signals": ["openrouter_free_model_rate_limit_error"], "provider_fields": [], "scope_note": "OpenRouter documents free-model quota as account-level; provider does not expose current free request usage."}',
    "openrouter_free_daily_local_meter",
    None,
    None,
    "call-fixed-1",
)

GOOGLE_QUOTA_OBSERVATION = {
    "observed_at": datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    "provider": "gemini",
    "client_family": "google_code_assist",
    "account_hash": None,
    "environment": None,
    "tenant_id": None,
    "repository": None,
    "session_id": None,
    "trace_id": None,
    "litellm_call_id": "call-fixed-1",
    "route_family": None,
    "request_model": None,
    "response_model": None,
    "model": "gemini-2.5-pro",
    "model_family": "gemini",
    "model_tier": "pro",
    "client_name": None,
    "client_version": None,
    "client_user_agent": None,
    "metadata": {},
    "source": "google_retrieve_user_quota",
    "limit_id": "google_code_assist_requests_gemini-2.5-pro",
    "limit_name": "Google Code Assist gemini-2.5-pro requests",
    "limit_scope": "model_requests",
    "window_minutes": 1440,
    "quota_period": "daily",
    "quota_type": None,
    "provider_resets_at": datetime(2026, 7, 2, 0, 0, 0, tzinfo=timezone.utc),
    "used_percentage": None,
    "remaining_requests": 80,
    "used_requests": None,
    "total_requests": 100,
    "raw_provider_fields": {
        "modelId": "gemini-2.5-pro",
        "remainingRequests": 80,
        "totalRequests": 100,
        "resetsAt": "2026-07-02T00:00:00Z",
    },
    "evidence": {
        "signals": [
            "google_quota_payload",
        ],
        "provider_fields": [
            "modelId",
            "remainingRequests",
            "resetsAt",
            "totalRequests",
        ],
        "token_type": None,
    },
    "inferred_window_start_at": datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
    "reset_hint_seconds": None,
    "limit_key": "gemini:google_code_assist:unknown_account:google_code_assist_requests_gemini-2.5-pro:model_requests:1440",
    "exhausted": False,
    "status": "observed",
}

GOOGLE_QUOTA_DB_PAYLOAD = (
    datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    "google_code_assist",
    None,
    None,
    "google",
    "gemini-2.5-pro",
    "google_code_assist_requests_gemini-2.5-pro:model_requests",
    "daily",
    "requests",
    datetime(2026, 7, 2, 0, 0, 0, tzinfo=timezone.utc),
    None,
    100.0,
    None,
    80.0,
    None,
    None,
    '{"modelId": "gemini-2.5-pro", "remainingRequests": 80, "totalRequests": 100, "resetsAt": "2026-07-02T00:00:00Z"}',
    '{"signals": ["google_quota_payload"], "provider_fields": ["modelId", "remainingRequests", "resetsAt", "totalRequests"], "token_type": null}',
    "google_retrieve_user_quota",
    None,
    None,
    "call-fixed-1",
)


# (case_id, input_builder, extractor, golden_observation, golden_db_payload)
_CASES = [
    (
        "codex_header",
        _codex_inputs,
        _extract_codex_header_rate_limit_observations,
        CODEX_OBSERVATION,
        CODEX_DB_PAYLOAD,
    ),
    (
        "anthropic_header",
        _anthropic_inputs,
        _extract_anthropic_header_rate_limit_observations,
        ANTHROPIC_OBSERVATION,
        ANTHROPIC_DB_PAYLOAD,
    ),
    (
        "xai_oauth_header",
        _xai_oauth_inputs,
        _extract_xai_oauth_header_rate_limit_observations,
        XAI_OAUTH_OBSERVATION,
        XAI_OAUTH_DB_PAYLOAD,
    ),
    (
        "grok_billing",
        _grok_billing_inputs,
        _extract_grok_billing_observations,
        GROK_BILLING_OBSERVATION,
        GROK_BILLING_DB_PAYLOAD,
    ),
    (
        "openrouter_free",
        _openrouter_free_inputs,
        _extract_openrouter_free_error_observations,
        OPENROUTER_FREE_OBSERVATION,
        OPENROUTER_FREE_DB_PAYLOAD,
    ),
    (
        "google_quota",
        _google_quota_inputs,
        _extract_google_quota_observations,
        GOOGLE_QUOTA_OBSERVATION,
        GOOGLE_QUOTA_DB_PAYLOAD,
    ),
]

_CASE_IDS = [row[0] for row in _CASES]


@pytest.mark.parametrize(
    "case_id,make_inputs,extractor,golden_observation,golden_db_payload",
    _CASES,
    ids=_CASE_IDS,
)
def test_extractor_observation_golden(case_id, make_inputs, extractor, golden_observation, golden_db_payload):
    """Each extractor maps its captured candidate to the exact observation.

    Pins the full record-dict output (every field) so the A3A extraction cannot
    silently alter any derived value -- limit_key, quota_period, window start,
    remaining/quota math, raw_provider_fields, or evidence.
    """
    kwargs, result = make_inputs()
    observations = extractor(kwargs, result, OBSERVED_AT)
    assert observations == [golden_observation]


@pytest.mark.parametrize(
    "case_id,make_inputs,extractor,golden_observation,golden_db_payload",
    _CASES,
    ids=_CASE_IDS,
)
def test_observation_dataclass_roundtrip_matches_legacy_dict(
    case_id, make_inputs, extractor, golden_observation, golden_db_payload
):
    """Observation -> `_build_rate_limit_observation_db_payload` is byte-identical.

    Drives the captured candidate through the current extraction path and then
    through the DB payload builder, asserting the resulting 22-tuple matches the
    pre-move golden. This is the storage parity contract Wave A3A must preserve:
    the record-dict input/output contract and the DB payload tuple shapes are
    UNCHANGED this wave.
    """
    kwargs, result = make_inputs()
    observations = extractor(kwargs, result, OBSERVED_AT)
    assert len(observations) == 1
    db_payload = _build_rate_limit_observation_db_payload(observations[0])
    assert db_payload == golden_db_payload
