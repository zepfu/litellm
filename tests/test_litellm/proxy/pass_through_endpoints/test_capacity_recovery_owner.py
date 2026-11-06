from types import SimpleNamespace

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.capacity_recovery import (
    CapacityRecoveryAction,
    CapacityRecoveryObservation,
    CapacityRecoveryOwner,
    get_or_create_capacity_recovery_owner,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.provider_call_ledger import (
    authorize_capacity_recovery_replay,
    is_openai_wire_replay_prohibited,
)


def _capacity_observation(**overrides):
    values = {
        "provider_returned": True,
        "status_code": 429,
        "error_class": "rate_limited",
        "retryable": True,
        "commitment": {
            "precommit_replay_committed": True,
            "precommit_commitment_reason": "provider_execution_possible",
        },
    }
    values.update(overrides)
    return CapacityRecoveryObservation(**values)


def test_owner_retries_once_then_transitions_to_another_route():
    owner = CapacityRecoveryOwner(max_same_route_retries=1)
    first = owner.observe(_capacity_observation())
    second = owner.observe(
        _capacity_observation(alternate_route_available=True)
    )
    assert first.action is CapacityRecoveryAction.RETRY_SAME_ROUTE
    assert second.action is CapacityRecoveryAction.ROUTE_TRANSITION
    assert second.retry_ordinal == 2


def test_owner_preserves_unknown_provider_outcome_as_replay_veto():
    owner = CapacityRecoveryOwner()
    decision = owner.observe(
        _capacity_observation(provider_returned=False, alternate_route_available=True)
    )
    assert decision.action is CapacityRecoveryAction.REPLAY_VETO
    assert decision.reason == "provider_outcome_unknown"


def test_owner_allows_typed_omitted_output_capacity_failure():
    owner = CapacityRecoveryOwner()
    decision = owner.observe(
        _capacity_observation(
            commitment={
                "precommit_replay_committed": True,
                "precommit_commitment_reason": "unknown_semantics",
            },
            error_payload={
                "type": "response.failed",
                "response": {
                    "id": "resp_1",
                    "status": "failed",
                    "error": {"code": "server_overloaded"},
                },
            },
        )
    )
    assert decision.action is CapacityRecoveryAction.RETRY_SAME_ROUTE


def test_owner_vetoes_continuation_and_invalid_ledger():
    owner = CapacityRecoveryOwner()
    continuation = owner.observe(_capacity_observation(continuation=True))
    invalid_ledger = owner.observe(_capacity_observation(ledger_valid=False))
    assert continuation.action is CapacityRecoveryAction.REPLAY_VETO
    assert invalid_ledger.action is CapacityRecoveryAction.REPLAY_VETO


def test_owner_is_request_scoped():
    request = SimpleNamespace(state=SimpleNamespace())
    first = get_or_create_capacity_recovery_owner(request)
    second = get_or_create_capacity_recovery_owner(request)
    assert first is second


def test_non_ledger_capacity_handoff_clears_only_dispatch_latch():
    request = SimpleNamespace(
        state=SimpleNamespace(
            _aawm_openai_responses_wire_commitment={
                "precommit_replay_committed": True,
                "precommit_commitment_reason": "provider_execution_possible",
                "response_start_sent": False,
                "first_body_sent": False,
                "terminal_wire_committed": False,
                "done_wire_committed": False,
            }
        )
    )
    authorize_capacity_recovery_replay(request)
    commitment = request.state._aawm_openai_responses_wire_commitment
    assert is_openai_wire_replay_prohibited(commitment, ledger=None) is False
    commitment["first_body_sent"] = True
    assert is_openai_wire_replay_prohibited(commitment, ledger=None) is True
