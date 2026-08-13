from datetime import datetime, timedelta, timezone

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection


_NOW = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
_NOW_EPOCH = _NOW.timestamp()


def _observation(
    *,
    model: str,
    quota_type: str,
    quota_period: str,
    remaining_pct: float,
    exhausted: bool,
    quota_used: object = None,
    observed_at: datetime = _NOW,
    expected_reset_at: datetime = _NOW + timedelta(hours=1),
) -> dict[str, object]:
    return {
        "provider": "cohere",
        "model": model,
        "quota_key": "cohere_trial_default",
        "quota_type": quota_type,
        "limit_scope": "credential",
        "quota_period": quota_period,
        "window_minutes": 1 if quota_type == "rpm" else None,
        "remaining_pct": remaining_pct,
        "observed_at": observed_at.isoformat(),
        "expected_reset_at": expected_reset_at.isoformat(),
        "status": "exhausted" if exhausted else "available",
        "exhausted": exhausted,
        "source": "locally_counted",
        "quota_used": quota_used,
    }


def _candidate(*, provider: str = "cohere", model: str = "cohere/command-a"):
    return {
        "provider": provider,
        "model": model,
        "route_family": "codex_cohere_chat_completions_adapter",
    }


def _state(candidate: dict[str, str]) -> dict[str, object]:
    lane_key = "cohere_native" if candidate["provider"] == "cohere" else "openrouter"
    return {
        "candidate": candidate,
        "lane_key": lane_key,
        "cooldown_seconds": 0.0,
    }


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    manager = selection.alias_routing_state
    manager.reset_for_tests()
    yield
    manager.reset_for_tests()


def _apply(
    candidate: dict[str, str],
    *,
    rpm: object = 20,
    observations: list[dict[str, object]],
) -> dict[str, object]:
    selection.alias_routing_state.record_normalized_quota_observations(observations)
    return selection._apply_cohere_local_quota_state(
        _state(candidate),
        now_epoch=_NOW_EPOCH,
    )


def test_should_not_block_monthly_below_numeric_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(selection, "get_model_info", lambda **_: {"rpm": 20})

    state = _apply(
        _candidate(model="cohere/command-b"),
        observations=[
            _observation(
                model="cohere/command-a",
                quota_type="monthly",
                quota_period="calendar_month",
                remaining_pct=0,
                exhausted=True,
                quota_used=999,
            )
        ],
    )

    assert "skip_reason" not in state
    assert selection._is_auto_agent_candidate_state_available(state)
    monthly = selection.alias_routing_state.resolve_cohere_monthly_observation(
        now_epoch=_NOW_EPOCH
    )
    assert monthly is not None
    assert {
        "provider",
        "model",
        "quota_key",
        "quota_type",
        "limit_scope",
        "quota_period",
        "window_minutes",
        "remaining_pct",
        "observed_at",
        "expected_reset_at",
        "status",
        "exhausted",
        "source",
    }.issubset(monthly)
    assert monthly["quota_used"] == 999.0


def test_should_block_monthly_at_numeric_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(selection, "get_model_info", lambda **_: {"rpm": 20})

    state = _apply(
        _candidate(model="cohere/command-b"),
        observations=[
            _observation(
                model="cohere/command-a",
                quota_type="monthly",
                quota_period="calendar_month",
                remaining_pct=100,
                exhausted=False,
                quota_used=1000,
            )
        ],
    )

    assert state["skip_reason"] == "quota_exhausted"
    assert not selection._is_auto_agent_candidate_state_available(state)


def test_should_expose_unknown_usage_as_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(selection, "get_model_info", lambda **_: {"rpm": 20})

    state = _apply(
        _candidate(model="cohere/command-a"),
        observations=[
            _observation(
                model="cohere/command-a",
                quota_type="monthly",
                quota_period="calendar_month",
                remaining_pct=0,
                exhausted=True,
            )
        ],
    )

    assert "skip_reason" not in state
    monthly = selection.alias_routing_state.resolve_cohere_monthly_observation(
        now_epoch=_NOW_EPOCH
    )
    assert monthly is not None
    assert monthly["quota_used"] is None


@pytest.mark.parametrize("quota_used", [19, 19.9])
def test_should_not_block_rpm_below_configured_limit(
    monkeypatch: pytest.MonkeyPatch,
    quota_used: float,
):
    monkeypatch.setattr(selection, "get_model_info", lambda **_: {"rpm": 20})

    state = _apply(
        _candidate(model="cohere/command-a"),
        observations=[
            _observation(
                model="cohere/command-a",
                quota_type="rpm",
                quota_period="rolling",
                remaining_pct=0,
                exhausted=True,
                quota_used=quota_used,
            )
        ],
    )

    assert "skip_reason" not in state
    assert selection._is_auto_agent_candidate_state_available(state)


def test_should_not_treat_boolean_rpm_metadata_as_numeric_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(selection, "get_model_info", lambda **_: {"rpm": True})

    state = _apply(
        _candidate(model="cohere/command-a"),
        observations=[
            _observation(
                model="cohere/command-a",
                quota_type="rpm",
                quota_period="rolling",
                remaining_pct=0,
                exhausted=True,
                quota_used=1,
            )
        ],
    )

    assert "skip_reason" not in state
    assert selection._is_auto_agent_candidate_state_available(state)


@pytest.mark.parametrize("quota_used", [20, 21])
def test_should_block_rpm_at_or_above_configured_limit(
    monkeypatch: pytest.MonkeyPatch,
    quota_used: float,
):
    monkeypatch.setattr(selection, "get_model_info", lambda **_: {"rpm": 20})

    state = _apply(
        _candidate(model="cohere/command-a"),
        observations=[
            _observation(
                model="cohere/command-a",
                quota_type="rpm",
                quota_period="rolling",
                remaining_pct=100,
                exhausted=False,
                quota_used=quota_used,
            )
        ],
    )

    assert state["skip_reason"] == "quota_exhausted"
    assert not selection._is_auto_agent_candidate_state_available(state)


def test_should_share_monthly_exhaustion_across_models(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(selection, "get_model_info", lambda **_: {"rpm": 20})

    state = _apply(
        _candidate(model="cohere/command-b"),
        observations=[
            _observation(
                model="cohere/command-a",
                quota_type="monthly",
                quota_period="calendar_month",
                remaining_pct=100,
                exhausted=False,
                quota_used=1000,
            )
        ],
    )

    assert state["skip_reason"] == "quota_exhausted"
    assert not selection._is_auto_agent_candidate_state_available(state)


def test_should_preserve_generic_unknown_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(selection, "get_model_info", lambda **_: {})

    _apply(
        _candidate(model="cohere/command-a"),
        observations=[
            _observation(
                model="cohere/command-a",
                quota_type="rpm",
                quota_period="rolling",
                remaining_pct=50,
                exhausted=False,
            )
        ],
    )
    rpm = selection.alias_routing_state.resolve_cohere_rpm_observation(
        model="cohere/command-a",
        now_epoch=_NOW_EPOCH,
    )
    assert rpm is not None
    assert rpm["quota_used"] is None
    assert rpm["expected_reset_at"] == _NOW_EPOCH + timedelta(hours=1).total_seconds()
    assert rpm["exhausted"] is None


def test_should_exhaust_exact_model_rpm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(selection, "get_model_info", lambda **_: {"rpm": 20})

    state = _apply(
        _candidate(model="cohere/command-a"),
        observations=[
            _observation(
                model="cohere/command-a",
                quota_type="monthly",
                quota_period="calendar_month",
                remaining_pct=50,
                exhausted=False,
            ),
            _observation(
                model="cohere/command-a",
                quota_type="rpm",
                quota_period="rolling",
                remaining_pct=0,
                exhausted=True,
                quota_used=20,
            ),
        ],
    )

    assert state["skip_reason"] == "quota_exhausted"
    assert not selection._is_auto_agent_candidate_state_available(state)


def test_should_leave_another_model_eligible(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(selection, "get_model_info", lambda **_: {"rpm": 20})

    state = _apply(
        _candidate(model="cohere/command-b"),
        observations=[
            _observation(
                model="cohere/command-a",
                quota_type="monthly",
                quota_period="calendar_month",
                remaining_pct=50,
                exhausted=False,
            ),
            _observation(
                model="cohere/command-a",
                quota_type="rpm",
                quota_period="rolling",
                remaining_pct=0,
                exhausted=True,
                quota_used=20,
            ),
        ],
    )

    assert "skip_reason" not in state
    assert selection._is_auto_agent_candidate_state_available(state)


def test_should_treat_missing_rpm_as_unknown(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(selection, "get_model_info", lambda **_: {})

    state = _apply(
        _candidate(model="cohere/command-a"),
        observations=[
            _observation(
                model="cohere/command-a",
                quota_type="monthly",
                quota_period="calendar_month",
                remaining_pct=50,
                exhausted=False,
            ),
            _observation(
                model="cohere/command-a",
                quota_type="rpm",
                quota_period="rolling",
                remaining_pct=0,
                exhausted=True,
                quota_used=20,
            ),
        ],
    )

    assert "skip_reason" not in state
    assert selection._is_auto_agent_candidate_state_available(state)


@pytest.mark.parametrize(
    "observed_at, expected_reset_at",
    [
        (_NOW - timedelta(hours=2), _NOW + timedelta(hours=1)),
        (_NOW, _NOW - timedelta(seconds=1)),
    ],
)
def test_should_ignore_stale_or_reset_monthly_observation(
    monkeypatch: pytest.MonkeyPatch,
    observed_at: datetime,
    expected_reset_at: datetime,
):
    monkeypatch.setattr(selection, "get_model_info", lambda **_: {"rpm": 20})

    state = _apply(
        _candidate(model="cohere/command-a"),
        observations=[
            _observation(
                model="cohere/command-a",
                quota_type="monthly",
                quota_period="calendar_month",
                remaining_pct=0,
                exhausted=True,
                observed_at=observed_at,
                expected_reset_at=expected_reset_at,
            )
        ],
    )

    assert "skip_reason" not in state
    assert selection._is_auto_agent_candidate_state_available(state)


def test_should_leave_openrouter_eligible(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(selection, "get_model_info", lambda **_: {"rpm": 20})

    state = _apply(
        _candidate(
            provider="openrouter",
            model="openrouter/cohere/north-mini-code:free",
        ),
        observations=[
            _observation(
                model="cohere/command-a",
                quota_type="monthly",
                quota_period="calendar_month",
                remaining_pct=0,
                exhausted=True,
            )
        ],
    )

    assert "skip_reason" not in state
    assert selection._is_auto_agent_candidate_state_available(state)
