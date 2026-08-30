"""OPENAI-006 encrypted-reasoning provenance and OpenAI egress compatibility."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    encrypted_reasoning_provenance as erp,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


CIPHERTEXT = "gAAAAABpnW_yEYmSNEyOG_ORIGINAL_BYTES_do_not_mutate=="
XAI_CIPHERTEXT = "gAAAAABpnW_xai_FOREIGN_REASONING_BYTES_xxxx=="


def _reasoning_item(
    *,
    item_id: str = "rs_1e1f2dad-631d-9e15-b4f3-0c8f60403798",
    encrypted_content: str = CIPHERTEXT,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "reasoning",
        "id": item_id,
        "encrypted_content": encrypted_content,
    }
    if provenance is not None:
        item[erp.PROVENANCE_ITEM_FIELD] = provenance
    return item


def test_compatibility_source_is_provider_family_plus_state_format():
    assert (
        erp.ENCRYPTED_REASONING_COMPATIBILITY_SOURCE
        == "producer_provider_family+encrypted_state_format"
    )


def test_same_provider_stamp_preserves_ciphertext_bytes():
    provenance = erp.build_encrypted_reasoning_provenance(
        producer_provider="openai",
        producer_model="gpt-5.6-sol",
        producer_route_family="codex_oauth",
        account_label="account1",
        account_lane="chatgpt-account:hash1",
        account_hash="hash1",
    )
    item = _reasoning_item()
    stamped = erp.stamp_encrypted_reasoning_provenance_on_item(item, provenance)
    assert stamped is item
    assert isinstance(item["encrypted_content"], str)
    assert item["encrypted_content"].startswith("aawm_erp:")
    _prov, original = erp.unwrap_encrypted_content_with_provenance(
        item["encrypted_content"]
    )
    assert original == CIPHERTEXT
    assert original.encode("utf-8") == CIPHERTEXT.encode("utf-8")
    assert _prov is not None
    assert _prov["producer_provider_family"] == "openai"
    assert (
        _prov["encrypted_state_format"]
        == erp.STATE_FORMAT_OPENAI_ENCRYPTED_REASONING
    )
    assert (
        _prov["compatibility_source"]
        == erp.ENCRYPTED_REASONING_COMPATIBILITY_SOURCE
    )
    # Sidecar present and safe (no ciphertext)
    sidecar = item[erp.PROVENANCE_ITEM_FIELD]
    assert CIPHERTEXT not in str(sidecar)
    assert "gAAAA" not in str(sidecar)


def test_prepare_egress_restores_same_provider_bytes_and_strips_sidecar():
    provenance = erp.build_encrypted_reasoning_provenance(
        producer_provider="openai",
        producer_model="gpt-5.6-sol",
        producer_route_family="codex_oauth",
        account_label="account1",
    )
    item = erp.stamp_encrypted_reasoning_provenance_on_item(
        _reasoning_item(), provenance
    )
    body = {
        "model": "gpt-5.6-sol",
        "input": [item, {"type": "message", "role": "user", "content": "hi"}],
    }
    prepared, disposition = erp.prepare_encrypted_reasoning_items_for_openai_egress(
        body
    )
    out_item = prepared["input"][0]
    assert out_item["encrypted_content"] == CIPHERTEXT
    assert out_item["encrypted_content"].encode("utf-8") == CIPHERTEXT.encode(
        "utf-8"
    )
    assert erp.PROVENANCE_ITEM_FIELD not in out_item
    assert disposition["encrypted_reasoning_item_count"] == 1
    assert disposition["encrypted_reasoning_disposition"] == "present"
    assert "openai" in disposition["encrypted_reasoning_producer_provider_families"]


def test_guard_allows_openai_same_provider_and_merges_disposition():
    provenance = erp.build_encrypted_reasoning_provenance(
        producer_provider="openai",
        producer_model="gpt-5.6-sol",
        producer_route_family="codex_oauth",
        account_label="account2",
        account_lane="lane-account2",
    )
    item = erp.stamp_encrypted_reasoning_provenance_on_item(
        _reasoning_item(), provenance
    )
    body = {"model": "gpt-5.6-sol", "input": [item]}
    prepared, disposition = erp.guard_openai_encrypted_reasoning_egress(body)
    assert prepared["input"][0]["encrypted_content"] == CIPHERTEXT
    assert disposition["encrypted_reasoning_compatible"] is True
    assert (
        disposition["encrypted_reasoning_disposition"] == "allowed_compatible"
    )
    merged = erp.merge_encrypted_reasoning_disposition_into_request_body(
        prepared, disposition
    )
    meta = merged["litellm_metadata"]
    assert meta["encrypted_reasoning_disposition"] == "allowed_compatible"
    assert "encrypted-reasoning:allowed_compatible" in meta["tags"]
    assert "encrypted-reasoning-producer:openai" in meta["tags"]
    # No ciphertext / bearer leakage in metadata
    dumped = str(meta)
    assert CIPHERTEXT not in dumped
    assert "Bearer " not in dumped


def test_guard_rejects_known_xai_foreign_before_upstream():
    provenance = erp.build_encrypted_reasoning_provenance(
        producer_provider="xai",
        producer_model="oa_xai/grok-4.5",
        producer_route_family="codex_xai_oauth_responses_adapter",
        account_label="xai_oauth_managed",
    )
    assert provenance["producer_provider_family"] == "xai"
    assert (
        provenance["encrypted_state_format"]
        == erp.STATE_FORMAT_XAI_ENCRYPTED_REASONING
    )
    item = erp.stamp_encrypted_reasoning_provenance_on_item(
        _reasoning_item(encrypted_content=XAI_CIPHERTEXT), provenance
    )
    body = {"model": "gpt-5.6-sol", "input": [item]}
    with pytest.raises(HTTPException) as exc_info:
        erp.guard_openai_encrypted_reasoning_egress(
            body,
            session_identity="session-abc",
            failure_phase="encrypted_reasoning_openai_pre_send",
        )
    err = exc_info.value
    assert err.status_code == 409
    detail = err.detail
    assert isinstance(detail, dict)
    assert detail["redispatch_required"] is True
    assert detail["non_resumable"] is True
    assert detail["fresh_dispatch_required"] is True
    assert detail["attempted_provider_call"] is False
    assert detail["error"]["code"] == "aawm_encrypted_reasoning_redispatch_required"
    assert "xai" in detail["redispatch_reason"]
    encrypted = detail["encrypted_reasoning"]
    assert encrypted["encrypted_reasoning_compatible"] is False
    assert (
        encrypted["encrypted_reasoning_disposition"]
        == "rejected_incompatible_producer"
    )
    assert XAI_CIPHERTEXT not in str(detail)


def test_pre_send_incompatible_reasoning_emits_one_sanitized_terminal_error():
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import audit_persist

    provenance = erp.build_encrypted_reasoning_provenance(
        producer_provider="xai",
        producer_model="producer-secret-model",
        producer_route_family="secret-route-family",
        account_label="secret-account",
    )
    item = erp.stamp_encrypted_reasoning_provenance_on_item(
        _reasoning_item(
            item_id="rs-secret-item",
            encrypted_content=XAI_CIPHERTEXT,
        ),
        provenance,
    )
    body = {
        "model": "gpt-5.6-sol",
        "input": [
            item,
            {
                "type": "message",
                "role": "user",
                "content": "prompt-secret-value",
            },
        ],
    }
    request = MagicMock()
    request.state = SimpleNamespace()

    class _SA:
        @staticmethod
        def resolve_canonical_session_identity(request, body):
            return "session-secret-identity"

    with patch.object(pte, "_session_affinity_mod", lambda: _SA), patch.object(
        audit_persist.verbose_proxy_logger,
        "error",
    ) as mock_error:
        with pytest.raises(HTTPException) as exc_info:
            pte._aawm_apply_openai_encrypted_reasoning_pre_send(
                request=request,
                parsed_body=body,
                custom_llm_provider="openai",
                egress_credential_family="codex_oauth",
                expected_target_family="openai",
                url=httpx.URL("https://chatgpt.com/backend-api/codex/responses"),
                provider_bound_body=body,
            )
        assert (
            pte._emit_openai_encrypted_reasoning_redispatch_terminal_error(
                exc_info.value,
                marker=request.state,
                correlation_id="outer-correlation-secret",
            )
            is False
        )

    error = exc_info.value
    assert error.status_code == 409
    detail = error.detail
    assert isinstance(detail, dict)
    assert detail["error"]["code"] == "aawm_encrypted_reasoning_redispatch_required"
    assert detail["failure_phase"] == "encrypted_reasoning_openai_pre_send"
    assert detail["redispatch_required"] is True
    assert detail["attempted_provider_call"] is False
    assert XAI_CIPHERTEXT not in str(detail)

    mock_error.assert_called_once()
    fields = mock_error.call_args.kwargs["extra"]
    assert fields["error_code"] == detail["error"]["code"]
    assert fields["status_code"] == error.status_code
    assert fields["failure_phase"] == detail["failure_phase"]
    assert fields["redispatch_required"] is True
    assert fields["attempted_provider_call"] is False
    assert fields["correlation_id"].startswith("sha256:")
    rendered = json.dumps(mock_error.call_args, default=str)
    for secret in (
        XAI_CIPHERTEXT,
        "rs-secret-item",
        "producer-secret-model",
        "secret-route-family",
        "secret-account",
        "prompt-secret-value",
        "session-secret-identity",
    ):
        assert secret not in rendered


def test_guard_rejects_foreign_state_format_even_if_provider_label_missing_family():
    # Explicit foreign state format must fail closed.
    item = _reasoning_item(
        provenance={
            "version": 1,
            "producer_provider_family": "kimi_code",
            "encrypted_state_format": erp.STATE_FORMAT_FOREIGN_ENCRYPTED_REASONING,
            "compatibility_source": erp.ENCRYPTED_REASONING_COMPATIBILITY_SOURCE,
        }
    )
    body = {"model": "gpt-5.6-sol", "input": [item]}
    with pytest.raises(HTTPException) as exc_info:
        erp.guard_openai_encrypted_reasoning_egress(body)
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["attempted_provider_call"] is False


def test_cross_account_openai_not_rejected_solely_due_to_account_lane():
    """Account1-produced encrypted content may continue on account2 lane."""
    producer = erp.build_encrypted_reasoning_provenance(
        producer_provider="openai",
        producer_model="gpt-5.6-sol",
        producer_route_family="codex_oauth",
        account_label="account1",
        account_lane="chatgpt-account:hash-account1",
        account_hash="hash-account1",
    )
    item = erp.stamp_encrypted_reasoning_provenance_on_item(
        _reasoning_item(), producer
    )
    body = {
        "model": "gpt-5.6-sol",
        "input": [item],
        "litellm_metadata": {
            "codex_auto_agent_selected_account_label": "account2",
            "codex_auto_agent_selected_account_lane": "chatgpt-account:hash-account2",
            "codex_auto_agent_selected_account_hash": "hash-account2",
        },
    }
    prepared, disposition = erp.guard_openai_encrypted_reasoning_egress(body)
    assert prepared["input"][0]["encrypted_content"] == CIPHERTEXT
    assert disposition["encrypted_reasoning_compatible"] is True
    # Producer account remains in item summary for audit; target account is not
    # a rejection reason.
    summaries = disposition["encrypted_reasoning_items"]
    assert summaries[0]["account_label"] == "account1"
    assert summaries[0]["account_lane"] == "chatgpt-account:hash-account1"


def test_unstamped_legacy_openai_items_remain_compatible():
    body = {
        "model": "gpt-5.6-sol",
        "input": [_reasoning_item()],  # no provenance
    }
    prepared, disposition = erp.guard_openai_encrypted_reasoning_egress(body)
    assert prepared["input"][0]["encrypted_content"] == CIPHERTEXT
    assert disposition["encrypted_reasoning_compatible"] is True


def test_safe_disposition_metadata_excludes_secrets_and_ciphertext():
    items = [
        {
            "producer_provider_family": "xai",
            "encrypted_state_format": erp.STATE_FORMAT_XAI_ENCRYPTED_REASONING,
            "producer_model": "oa_xai/grok-4.5",
            "item_id": "rs_abc",
        }
    ]
    meta = erp.build_encrypted_reasoning_disposition_metadata(
        disposition="rejected_incompatible_producer",
        items=items,
        mismatch_reason="incompatible_encrypted_reasoning_producer:xai",
        compatibility_ok=False,
    )
    blob = str(meta)
    assert "gAAAA" not in blob
    assert "Bearer" not in blob
    assert meta["encrypted_reasoning_compatibility_source"] == (
        erp.ENCRYPTED_REASONING_COMPATIBILITY_SOURCE
    )


@pytest.mark.asyncio
async def test_pre_send_guard_rejects_xai_without_upstream_call(monkeypatch):
    """Shared pass_through pre-send seam fails closed before provider I/O."""
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte

    # Minimal session-affinity stub for identity only.
    class _SA:
        @staticmethod
        def resolve_canonical_session_identity(request, body):
            return "session-test"

        @staticmethod
        def request_session_owner_already_guarded(request):
            return True  # still must run encrypted-reasoning guard

        @staticmethod
        def get_request_session_owner_lease(request):
            return None

    monkeypatch.setattr(pte, "_session_affinity_mod", lambda: _SA)

    provenance = erp.build_encrypted_reasoning_provenance(
        producer_provider="xai",
        producer_model="oa_xai/grok-4.5",
        producer_route_family="codex_xai_oauth_responses_adapter",
    )
    item = erp.stamp_encrypted_reasoning_provenance_on_item(
        _reasoning_item(encrypted_content=XAI_CIPHERTEXT), provenance
    )
    body = {"model": "gpt-5.6-sol", "input": [item]}
    request = MagicMock()
    request.state = SimpleNamespace()
    url = SimpleNamespace(path="/v1/responses")

    with pytest.raises(HTTPException) as exc_info:
        await pte._aawm_session_owner_pre_send_guard(
            request=request,
            parsed_body=body,
            custom_llm_provider="openai",
            egress_credential_family="codex_oauth",
            expected_target_family="openai",
            url=url,
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["attempted_provider_call"] is False
    assert exc_info.value.detail["redispatch_required"] is True


@pytest.mark.asyncio
async def test_pre_send_guard_allows_cross_account_openai(monkeypatch):
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte

    class _SA:
        @staticmethod
        def resolve_canonical_session_identity(request, body):
            return "session-test"

        @staticmethod
        def request_session_owner_already_guarded(request):
            return True

        @staticmethod
        def get_request_session_owner_lease(request):
            return None

        @staticmethod
        def should_skip_session_owner_for_openai_models_discovery(
            request=None, *, endpoint=None, url=None
        ):
            return False

    monkeypatch.setattr(pte, "_session_affinity_mod", lambda: _SA)

    provenance = erp.build_encrypted_reasoning_provenance(
        producer_provider="openai",
        producer_model="gpt-5.6-sol",
        producer_route_family="codex_oauth",
        account_label="account1",
        account_lane="lane1",
        account_hash="hash1",
    )
    item = erp.stamp_encrypted_reasoning_provenance_on_item(
        _reasoning_item(), provenance
    )
    body = {
        "model": "gpt-5.6-sol",
        "input": [item],
        "litellm_metadata": {
            "codex_auto_agent_selected_account_label": "account2",
            "codex_auto_agent_selected_account_lane": "lane2",
        },
    }
    request = MagicMock()
    request.state = SimpleNamespace()
    url = SimpleNamespace(path="/backend-api/codex/responses")

    await pte._aawm_session_owner_pre_send_guard(
        request=request,
        parsed_body=body,
        custom_llm_provider="openai",
        egress_credential_family="codex_oauth",
        expected_target_family="openai",
        url=url,
    )
    # Ciphertext restored for egress
    assert body["input"][0]["encrypted_content"] == CIPHERTEXT
    assert erp.PROVENANCE_ITEM_FIELD not in body["input"][0]
    assert erp.ROUTE_IDENTITY_FIELD not in body["input"][0]
    meta = body["litellm_metadata"]
    assert meta["encrypted_reasoning_disposition"] == "allowed_compatible"
    assert getattr(request.state, "_aawm_encrypted_reasoning_disposition", None)


def test_strip_route_identity_from_request_body_drops_unknown_item_field():
    body = {
        "model": "provider-kimi_code",
        "input": [
            {
                "type": "message",
                "id": "msg_kimi_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "PONG"}],
                erp.ROUTE_IDENTITY_FIELD: {
                    "producer_provider": "kimi_code",
                    "producer_model": "kimi_code/k3",
                    "producer_route_family": "codex_kimi_chat_completions_adapter",
                },
            }
        ],
        erp.ROUTE_IDENTITY_FIELD: {
            "producer_provider": "kimi_code",
            "producer_model": "kimi_code/k3",
            "producer_route_family": "codex_kimi_chat_completions_adapter",
        },
    }
    stripped = erp.strip_route_identity_from_request_body(body)
    assert erp.ROUTE_IDENTITY_FIELD not in stripped
    assert erp.ROUTE_IDENTITY_FIELD not in stripped["input"][0]
    assert stripped["input"][0]["type"] == "message"
    assert stripped["input"][0]["content"][0]["text"] == "PONG"
    prepared, _disposition = erp.prepare_encrypted_reasoning_items_for_openai_egress(
        body
    )
    assert erp.ROUTE_IDENTITY_FIELD not in prepared
    assert erp.ROUTE_IDENTITY_FIELD not in prepared["input"][0]


@pytest.mark.asyncio
async def test_pre_send_guard_strips_route_identity_for_xai_responses(monkeypatch):
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte

    class _SA:
        @staticmethod
        def resolve_canonical_session_identity(request, body):
            return "session-test"

        @staticmethod
        def request_session_owner_already_guarded(request):
            return True

        @staticmethod
        def get_request_session_owner_lease(request):
            return None

        @staticmethod
        def should_skip_session_owner_for_openai_models_discovery(
            request=None, *, endpoint=None, url=None
        ):
            return False

    monkeypatch.setattr(pte, "_session_affinity_mod", lambda: _SA)

    identity = {
        "producer_provider": "xai",
        "producer_model": "oa_xai/grok-4.6",
        "producer_route_family": "codex_xai_oauth_responses_adapter",
    }
    body = {
        "model": "provider-xai",
        "input": [
            {
                "type": "message",
                "id": "msg_xai_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "PONG"}],
                erp.ROUTE_IDENTITY_FIELD: dict(identity),
            }
        ],
        erp.ROUTE_IDENTITY_FIELD: dict(identity),
    }
    request = MagicMock()
    request.state = SimpleNamespace()
    url = SimpleNamespace(path="/v1/responses")

    await pte._aawm_session_owner_pre_send_guard(
        request=request,
        parsed_body=body,
        custom_llm_provider="xai",
        egress_credential_family="xai",
        expected_target_family="xai",
        url=url,
    )
    assert erp.ROUTE_IDENTITY_FIELD not in body
    assert erp.ROUTE_IDENTITY_FIELD not in body["input"][0]
    assert body["input"][0]["content"][0]["text"] == "PONG"


def test_streaming_sse_stamp_preserves_ciphertext():
    event = {
        "type": "response.output_item.done",
        "item": {
            "type": "reasoning",
            "id": "rs_stream_1",
            "encrypted_content": CIPHERTEXT,
        },
    }

    chunk = f"data: {json.dumps(event)}\n\n".encode("utf-8")
    out = PassThroughStreamingHandler._stamp_encrypted_reasoning_in_responses_sse_chunk(
        chunk,
        request_body={"model": "gpt-5.6-sol"},
        custom_llm_provider="openai",
    )
    assert out != chunk
    text = out.decode("utf-8")
    assert "aawm_erp:" in text
    # Original ciphertext still embedded after wrap separator
    assert CIPHERTEXT in text
    # No double-encoding of whole event as ciphertext field only
    line = [ln for ln in text.splitlines() if ln.startswith("data:")][0]
    parsed = json.loads(line[len("data:") :].strip())
    enc = parsed["item"]["encrypted_content"]
    _p, original = erp.unwrap_encrypted_content_with_provenance(enc)
    assert original == CIPHERTEXT
    assert parsed["item"][erp.PROVENANCE_ITEM_FIELD]["producer_provider_family"] == (
        "openai"
    )


@pytest.mark.asyncio
async def test_fragmented_streaming_sse_stamp_happens_after_reassembly():
    event = {
        "type": "response.output_item.done",
        "item": {
            "type": "reasoning",
            "id": "rs_stream_split",
            "encrypted_content": CIPHERTEXT,
        },
    }
    reasoning_chunk = (
        "event: response.output_item.done\n"
        "data: "
        + json.dumps(event)
        + "\n\n"
    ).encode("utf-8")
    split_at = reasoning_chunk.index(b'"encrypted_content"') + len(
        b'"encrypted_content"'
    )
    terminal_chunk = (
        "event: response.completed\n"
        "data: "
        + json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_stream_split",
                    "status": "completed",
                    "output": [],
                },
            }
        )
        + "\n\n"
    ).encode("utf-8")
    provider_chunks = [
        reasoning_chunk[:split_at],
        reasoning_chunk[split_at:],
        terminal_chunk,
        b"data: [DONE]\n\n",
    ]

    async def _aiter_bytes():
        for provider_chunk in provider_chunks:
            yield provider_chunk

    response = MagicMock()
    response.headers = httpx.Headers({})
    response.aiter_bytes = _aiter_bytes
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj._update_completion_start_time = MagicMock()
    success_handler_kwargs = {
        "litellm_params": {
            "metadata": {
                "codex_auto_agent_selected_provider": "openai",
                "codex_auto_agent_selected_model": "gpt-5.6-sol",
                "codex_auto_agent_selected_route_family": "codex_oauth",
            }
        }
    }
    route_handler = AsyncMock()

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        route_handler,
    ):
        emitted = [
            chunk
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=response,
                request_body={
                    "model": "gpt-5.6-sol",
                    "litellm_metadata": {
                        "codex_auto_agent_selected_route_family": "codex_oauth",
                    },
                },
                litellm_logging_obj=logging_obj,
                endpoint_type=EndpointType.OPENAI,
                start_time=datetime.now(),
                passthrough_success_handler_obj=MagicMock(
                    spec=PassThroughEndpointLogging
                ),
                url_route="https://chatgpt.com/backend-api/codex/responses",
                custom_llm_provider="openai",
                success_handler_kwargs=success_handler_kwargs,
            )
        ]
        await asyncio.sleep(0.05)

    stamped_chunk = next(
        chunk for chunk in emitted if b"response.output_item.done" in chunk
    )
    data_line = next(
        line for line in stamped_chunk.splitlines() if line.startswith(b"data:")
    )
    parsed = json.loads(data_line.removeprefix(b"data:").strip())
    stamped_item = parsed["item"]
    assert stamped_item["encrypted_content"].startswith("aawm_erp:")
    _provenance, original = erp.unwrap_encrypted_content_with_provenance(
        stamped_item["encrypted_content"]
    )
    assert original == CIPHERTEXT
    assert stamped_item[erp.PROVENANCE_ITEM_FIELD][
        "producer_provider_family"
    ] == "openai"
    assert stamped_item[erp.ROUTE_IDENTITY_FIELD] == {
        "producer_provider": "openai",
        "producer_model": "gpt-5.6-sol",
        "producer_route_family": "codex_oauth",
    }
    route_handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_xai_fragmented_streaming_sse_stamps_after_reassembly():
    event = {
        "type": "response.output_item.done",
        "item": {
            "type": "reasoning",
            "id": "rs_xai_stream_split",
            "encrypted_content": XAI_CIPHERTEXT,
        },
    }
    reasoning_chunk = (
        "event: response.output_item.done\n"
        "data: "
        + json.dumps(event)
        + "\n\n"
    ).encode("utf-8")
    ciphertext_bytes = XAI_CIPHERTEXT.encode("utf-8")
    ciphertext_start = reasoning_chunk.index(ciphertext_bytes)
    split_at = ciphertext_start + len(ciphertext_bytes) // 2
    terminal_chunk = (
        "event: response.completed\n"
        "data: "
        + json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_xai_stream_split",
                    "status": "completed",
                    "output": [],
                },
            }
        )
        + "\n\n"
    ).encode("utf-8")
    provider_chunks = [
        reasoning_chunk[:split_at],
        reasoning_chunk[split_at:],
        terminal_chunk,
        b"data: [DONE]\n\n",
    ]

    async def _aiter_bytes():
        for provider_chunk in provider_chunks:
            yield provider_chunk

    response = MagicMock()
    response.headers = httpx.Headers({})
    response.aiter_bytes = _aiter_bytes
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj._update_completion_start_time = MagicMock()
    success_handler_kwargs = {
        "litellm_params": {
            "metadata": {
                "codex_auto_agent_selected_provider": "xai",
                "codex_auto_agent_selected_model": "grok-4.6",
                "codex_auto_agent_selected_route_family": "xai_oauth",
            }
        }
    }
    route_handler = AsyncMock()

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        route_handler,
    ):
        emitted = [
            chunk
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=response,
                request_body={
                    "model": "grok-4.6",
                    "litellm_metadata": {
                        "codex_auto_agent_selected_provider": "xai",
                        "codex_auto_agent_selected_model": "grok-4.6",
                        "codex_auto_agent_selected_route_family": "xai_oauth",
                    },
                },
                litellm_logging_obj=logging_obj,
                endpoint_type=EndpointType.OPENAI,
                start_time=datetime.now(),
                passthrough_success_handler_obj=MagicMock(
                    spec=PassThroughEndpointLogging
                ),
                url_route="https://api.x.ai/v1/responses",
                custom_llm_provider="xai",
                success_handler_kwargs=success_handler_kwargs,
            )
        ]
        await asyncio.sleep(0.05)

    stamped_chunk = next(
        chunk for chunk in emitted if b"response.output_item.done" in chunk
    )
    data_line = next(
        line for line in stamped_chunk.splitlines() if line.startswith(b"data:")
    )
    parsed = json.loads(data_line.removeprefix(b"data:").strip())
    stamped_item = parsed["item"]
    assert stamped_item["encrypted_content"].startswith("aawm_erp:")
    assert XAI_CIPHERTEXT in stamped_item["encrypted_content"]
    _provenance, original = erp.unwrap_encrypted_content_with_provenance(
        stamped_item["encrypted_content"]
    )
    assert original == XAI_CIPHERTEXT
    assert stamped_item[erp.ROUTE_IDENTITY_FIELD] == {
        "producer_provider": "xai",
        "producer_model": "grok-4.6",
        "producer_route_family": "xai_oauth",
    }
    route_handler.assert_awaited_once()


def _assert_clean_upstream_encrypted_item(sent_json: dict[str, Any], ciphertext: str) -> None:
    assert isinstance(sent_json, dict)
    input_items = sent_json.get("input")
    assert isinstance(input_items, list)
    reasoning = next(
        item
        for item in input_items
        if isinstance(item, dict) and item.get("type") == "reasoning"
    )
    enc = reasoning["encrypted_content"]
    assert enc == ciphertext
    assert enc.encode("utf-8") == ciphertext.encode("utf-8")
    assert not str(enc).startswith("aawm_erp:")
    assert "aawm_erp:" not in str(sent_json)
    assert erp.PROVENANCE_ITEM_FIELD not in reasoning
    assert erp.PROVENANCE_ITEM_FIELD not in str(sent_json)
    assert erp.ROUTE_IDENTITY_FIELD not in reasoning
    assert erp.ROUTE_IDENTITY_FIELD not in str(sent_json)


async def _run_pass_through_and_capture_json(
    *,
    stream: bool,
    custom_body: dict[str, Any],
    custom_llm_provider: str = "openai",
    egress_credential_family: str = "codex_oauth",
    expected_target_family: str = "openai",
) -> dict[str, Any]:
    """Drive pass_through_request and return the exact upstream JSON body."""
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        pass_through_request,
    )

    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url = "http://localhost:4000/openai/v1/responses"
    mock_request.headers = {"content-type": "application/json"}
    mock_request.query_params = {}
    mock_request.state = SimpleNamespace()

    mock_httpx_response = MagicMock()
    mock_httpx_response.status_code = 200
    mock_httpx_response.headers = {"content-type": "application/json"}
    if stream:
        mock_httpx_response.headers = {"content-type": "text/event-stream"}
    async def _aiter_bytes():
        yield b'data: {"type":"response.completed"}\n\n'

    mock_httpx_response.aiter_bytes = MagicMock(return_value=_aiter_bytes())
    mock_httpx_response.aread = AsyncMock(return_value=b'{"id":"resp_test","output":[]}')
    mock_httpx_response.raise_for_status = MagicMock()

    # Avoid real session-owner Redis/store during integration-style capture tests.
    class _SA:
        @staticmethod
        def resolve_canonical_session_identity(request, body):
            return "session-capture-test"

        @staticmethod
        def request_session_owner_already_guarded(request):
            return True

        @staticmethod
        def get_request_session_owner_lease(request):
            return None

        @staticmethod
        def should_skip_session_owner_for_openai_models_discovery(
            request=None, *, endpoint=None, url=None
        ):
            return False

        @staticmethod
        async def ensure_session_owner_guard_for_request(**kwargs):
            return None

        @staticmethod
        async def finalize_request_session_owner_lease(*args, **kwargs):
            return None

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
    ) as mock_get_client, patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ) as mock_logging_obj, patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
        new_callable=AsyncMock,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints._session_affinity_mod",
        lambda: _SA,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.validate_outgoing_egress",
        return_value=None,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.emit_aawm_route_access_log",
        return_value=None,
    ):
        mock_client = MagicMock()
        mock_client.build_request.return_value = MagicMock()
        mock_client.send = AsyncMock(return_value=mock_httpx_response)
        # non-stream path may use non_streaming_http_request_handler
        mock_client.post = AsyncMock(return_value=mock_httpx_response)
        mock_client.request = AsyncMock(return_value=mock_httpx_response)
        mock_client_obj = MagicMock()
        mock_client_obj.client = mock_client
        mock_get_client.return_value = mock_client_obj

        mock_logging_obj.pre_call_hook = AsyncMock(return_value=custom_body)
        mock_logging_obj.post_call_success_hook = AsyncMock()
        mock_logging_obj.post_call_failure_hook = AsyncMock()

        # Patch non-stream handler to capture body when used.
        from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte

        captured: dict[str, Any] = {}

        async def _capture_non_stream(**kwargs):
            body = kwargs.get("_parsed_body")
            if isinstance(body, dict):
                captured["json"] = body
            return mock_httpx_response

        with patch.object(
            pte.HttpPassThroughEndpointHelpers,
            "non_streaming_http_request_handler",
            new=AsyncMock(side_effect=_capture_non_stream),
        ):
            await pass_through_request(
                request=mock_request,
                target="https://api.openai.com/v1/responses",
                custom_headers={},
                user_api_key_dict=MagicMock(),
                custom_body=custom_body,
                stream=stream,
                custom_llm_provider=custom_llm_provider,
                egress_credential_family=egress_credential_family,
                expected_target_family=expected_target_family,
            )

        if mock_client.build_request.called:
            kwargs = mock_client.build_request.call_args.kwargs
            if "json" in kwargs and isinstance(kwargs["json"], dict):
                return kwargs["json"]
        if "json" in captured:
            return captured["json"]
        raise AssertionError("No upstream JSON body was captured")


@pytest.mark.asyncio
async def test_pass_through_nonstream_sends_unwrapped_ciphertext_byte_for_byte():
    """Non-stream provider_bound_body must strip aawm_erp wrap/sidecar before send."""
    provenance = erp.build_encrypted_reasoning_provenance(
        producer_provider="openai",
        producer_model="gpt-5.6-sol",
        producer_route_family="codex_oauth",
        account_label="account1",
    )
    item = erp.stamp_encrypted_reasoning_provenance_on_item(
        _reasoning_item(), provenance
    )
    assert item["encrypted_content"].startswith("aawm_erp:")
    custom_body = {
        "model": "gpt-5.6-sol",
        "input": [item, {"type": "message", "role": "user", "content": "hi"}],
        "litellm_metadata": {"session_id": "session-abc"},
    }
    sent = await _run_pass_through_and_capture_json(stream=False, custom_body=custom_body)
    _assert_clean_upstream_encrypted_item(sent, CIPHERTEXT)
    # litellm_metadata stripped from provider-bound body
    assert "litellm_metadata" not in sent


@pytest.mark.asyncio
async def test_pass_through_stream_sends_unwrapped_ciphertext_byte_for_byte():
    """Streaming send path must use the same normalized provider_bound_body."""
    provenance = erp.build_encrypted_reasoning_provenance(
        producer_provider="openai",
        producer_model="gpt-5.6-sol",
        producer_route_family="codex_oauth",
        account_label="account2",
        account_lane="lane-account2",
    )
    item = erp.stamp_encrypted_reasoning_provenance_on_item(
        _reasoning_item(), provenance
    )
    assert "aawm_erp:" in item["encrypted_content"]
    custom_body = {
        "model": "gpt-5.6-sol",
        "input": [item],
        "stream": True,
        "litellm_metadata": {
            "session_id": "session-stream",
            "codex_auto_agent_selected_account_label": "account2",
        },
    }
    sent = await _run_pass_through_and_capture_json(stream=True, custom_body=custom_body)
    _assert_clean_upstream_encrypted_item(sent, CIPHERTEXT)
    assert "litellm_metadata" not in sent


def _apply_openai_encrypted_reasoning_pre_send(
    *,
    body: dict[str, Any],
    url: Any,
    custom_llm_provider: str = "openai",
    egress_credential_family: str = "codex_oauth",
    expected_target_family: str = "openai",
) -> dict[str, Any]:
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte

    class _SA:
        @staticmethod
        def resolve_canonical_session_identity(request, body):
            return "session-test"

        @staticmethod
        def request_session_owner_already_guarded(request):
            return True

        @staticmethod
        def get_request_session_owner_lease(request):
            return None

    request = MagicMock()
    request.state = SimpleNamespace()
    with patch.object(pte, "_session_affinity_mod", lambda: _SA):
        pte._aawm_apply_openai_encrypted_reasoning_pre_send(
            request=request,
            parsed_body=body,
            custom_llm_provider=custom_llm_provider,
            egress_credential_family=egress_credential_family,
            expected_target_family=expected_target_family,
            url=url,
            provider_bound_body=body,
        )
    return body


def _wrapped_openai_function_output_ciphertext() -> str:
    return erp.wrap_encrypted_content_with_provenance(
        CIPHERTEXT,
        erp.build_encrypted_reasoning_provenance(
            producer_provider="openai",
            producer_model="gpt-5.6-sol",
            producer_route_family="codex_oauth",
        ),
    )


def test_pre_send_strips_ciphertext_only_function_output_on_chatgpt_unlabeled():
    """ChatGPT-host unlabeled Responses egress must drop ciphertext-only function output."""
    body = {
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_chatgpt_unlabeled_child",
                "encrypted_content": _wrapped_openai_function_output_ciphertext(),
            }
        ],
    }
    sent = _apply_openai_encrypted_reasoning_pre_send(
        body=body,
        url=SimpleNamespace(
            host="chatgpt.com",
            path="/backend-api/codex/responses",
        ),
        egress_credential_family="codex_oauth",
    )
    item = sent["input"][0]
    assert item["call_id"] == "call_chatgpt_unlabeled_child"
    assert "encrypted_content" not in item
    assert item.get("output") == ""
    assert CIPHERTEXT not in str(sent)


def test_pre_send_strips_ciphertext_only_function_output_on_chatgpt_unlabeled_gpt56_sol_child():
    """Live unlabeled gpt-5.6-sol ChatGPT-host child must still enter sanitation."""
    body = {
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_chatgpt_unlabeled_gpt56_sol_child",
                "encrypted_content": _wrapped_openai_function_output_ciphertext(),
            }
        ],
    }
    sent = _apply_openai_encrypted_reasoning_pre_send(
        body=body,
        url=SimpleNamespace(
            host="chatgpt.com",
            path="/backend-api/codex/responses",
        ),
        custom_llm_provider=None,
        egress_credential_family=None,
        expected_target_family=None,
    )
    item = sent["input"][0]
    assert item["call_id"] == "call_chatgpt_unlabeled_gpt56_sol_child"
    assert "encrypted_content" not in item
    assert item.get("output") == ""
    assert CIPHERTEXT not in str(sent)


def test_pre_send_skips_unlabeled_chatgpt_lookalike_responses_path():
    """Unlabeled confirmed-host lookalike paths must not enter sanitation."""
    ciphertext = _wrapped_openai_function_output_ciphertext()
    body = {
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_chatgpt_unlabeled_lookalike",
                "encrypted_content": ciphertext,
            }
        ],
    }
    sent = _apply_openai_encrypted_reasoning_pre_send(
        body=body,
        url=SimpleNamespace(
            host="chatgpt.com",
            path="/backend-api/codex/responses-invalid",
        ),
        custom_llm_provider=None,
        egress_credential_family=None,
        expected_target_family=None,
    )
    item = sent["input"][0]
    assert item["call_id"] == "call_chatgpt_unlabeled_lookalike"
    assert item["encrypted_content"] == ciphertext


def test_pre_send_preserves_plaintext_function_output():
    """Valid plaintext function output must survive OpenAI Responses egress sanitation."""
    body = {
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_child_plaintext",
                "output": "Mon Aug 24 11:44:41 EDT 2026",
                "encrypted_content": _wrapped_openai_function_output_ciphertext(),
            }
        ],
    }
    sent = _apply_openai_encrypted_reasoning_pre_send(
        body=body,
        url=SimpleNamespace(
            host="chatgpt.com",
            path="/backend-api/codex/responses",
        ),
    )
    item = sent["input"][0]
    assert item["output"] == "Mon Aug 24 11:44:41 EDT 2026"
    assert "encrypted_content" not in item
    assert CIPHERTEXT not in str(sent)


def test_pre_send_strips_ciphertext_only_function_output_on_api_openai():
    """Explicit api.openai.com API-key Responses egress still strips ciphertext-only blobs."""
    body = {
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_api_key_child",
                "encrypted_content": _wrapped_openai_function_output_ciphertext(),
            }
        ],
    }
    sent = _apply_openai_encrypted_reasoning_pre_send(
        body=body,
        url=SimpleNamespace(host="api.openai.com", path="/v1/responses"),
        egress_credential_family="openai",
    )
    item = sent["input"][0]
    assert item["call_id"] == "call_api_key_child"
    assert "encrypted_content" not in item
    assert item.get("output") == ""
    assert CIPHERTEXT not in str(sent)


@pytest.mark.asyncio
async def test_native_openai_owner_strips_ciphertext_only_function_output_before_pass_through():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _perform_codex_auto_agent_native_openai_request,
    )

    captured: dict[str, Any] = {}

    async def _capture_pass_through(**kwargs):
        captured["body"] = kwargs.get("custom_body")
        return MagicMock()

    request_body = {
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_native_owner_child",
                "encrypted_content": "ciphertext",
            },
        ],
    }
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
        new=AsyncMock(side_effect=_capture_pass_through),
    ):
        await _perform_codex_auto_agent_native_openai_request(
            request=MagicMock(spec=Request),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            target_url="https://chatgpt.com/backend-api/codex/responses",
            api_key=None,
            forward_headers=True,
            request_body=request_body,
            custom_headers={},
        )
    sent = captured["body"]
    item = sent["input"][0]
    assert item["call_id"] == "call_native_owner_child"
    assert "encrypted_content" not in item


_CODEX_EMPTY_SPAWN_ENVELOPE = (
    "Message Type: NEW_TASK\n"
    "Task name: /root/hv2_child_shell_retry\n"
    "Sender: /root\n"
    "Payload:\n"
)
_CODEX_MESSAGE_SPAWN_ENVELOPE = (
    "Message Type: MESSAGE\n"
    "Task name: /root/hv2_child_shell_retry\n"
    "Sender: /root\n"
    "Payload:\n"
)
_CODEX_SPAWN_TASK_TEXT = (
    "gAAAA ordinary spawn task text; execute a harmless local shell command."
)


def test_guard_restores_codex_spawn_agent_plaintext_into_visible_input_text():
    """Native OpenAI egress must surface stranded spawn_agent task text."""
    body = {
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "agent_message",
                "content": [
                    {
                        "type": "input_text",
                        "text": _CODEX_EMPTY_SPAWN_ENVELOPE,
                    },
                    {
                        "type": "encrypted_content",
                        "encrypted_content": _CODEX_SPAWN_TASK_TEXT,
                    },
                ],
            }
        ],
    }
    prepared, _disposition = erp.guard_openai_encrypted_reasoning_egress(
        body,
        url="https://chatgpt.com/backend-api/codex/responses",
    )
    item = prepared["input"][0]
    assert item["type"] == "agent_message"
    assert item["content"] == [
        {
            "type": "input_text",
            "text": f"{_CODEX_EMPTY_SPAWN_ENVELOPE}{_CODEX_SPAWN_TASK_TEXT}",
        }
    ]


def test_guard_leaves_nonmatching_agent_message_shape_unmodified():
    """Only the exact NEW_TASK/input_text two-part wire shape is restored."""
    original_item = {
        "type": "agent_message",
        "content": [
            {
                "type": "input_text",
                "text": _CODEX_MESSAGE_SPAWN_ENVELOPE,
            },
            {
                "type": "encrypted_content",
                "encrypted_content": _CODEX_SPAWN_TASK_TEXT,
            },
        ],
    }
    body = {
        "model": "gpt-5.6-sol",
        "input": [original_item],
    }
    prepared, _disposition = erp.guard_openai_encrypted_reasoning_egress(
        body,
        url="https://chatgpt.com/backend-api/codex/responses",
    )
    item = prepared["input"][0]
    assert item == original_item
    assert item["content"] == [
        {
            "type": "input_text",
            "text": _CODEX_MESSAGE_SPAWN_ENVELOPE,
        },
        {
            "type": "encrypted_content",
            "encrypted_content": _CODEX_SPAWN_TASK_TEXT,
        },
    ]


@pytest.mark.asyncio
async def test_native_openai_owner_restores_spawn_agent_plaintext_and_strips_function_output():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _perform_codex_auto_agent_native_openai_request,
    )

    captured: dict[str, Any] = {}

    async def _capture_pass_through(**kwargs):
        captured["body"] = kwargs.get("custom_body")
        return MagicMock()

    request_body = {
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "agent_message",
                "content": [
                    {
                        "type": "input_text",
                        "text": _CODEX_EMPTY_SPAWN_ENVELOPE,
                    },
                    {
                        "type": "encrypted_content",
                        "encrypted_content": _CODEX_SPAWN_TASK_TEXT,
                    },
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call_native_owner_child",
                "encrypted_content": "ciphertext",
            },
        ],
    }
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
        new=AsyncMock(side_effect=_capture_pass_through),
    ):
        await _perform_codex_auto_agent_native_openai_request(
            request=MagicMock(spec=Request),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            target_url="https://chatgpt.com/backend-api/codex/responses",
            api_key=None,
            forward_headers=True,
            request_body=request_body,
            custom_headers={},
        )
    sent = captured["body"]
    agent_item = sent["input"][0]
    function_item = sent["input"][1]
    assert agent_item["type"] == "agent_message"
    assert agent_item["content"] == [
        {
            "type": "input_text",
            "text": f"{_CODEX_EMPTY_SPAWN_ENVELOPE}{_CODEX_SPAWN_TASK_TEXT}",
        }
    ]
    assert function_item["call_id"] == "call_native_owner_child"
    assert "encrypted_content" not in function_item


def test_build_route_identity_from_valid_provenance_triple():
    identity = erp.build_route_identity_from_provenance(
        {
            "producer_provider": "kimi_code",
            "producer_model": "kimi_code/k3",
            "producer_route_family": "codex_kimi_chat_completions_adapter",
        }
    )
    assert identity == {
        "producer_provider": "kimi_code",
        "producer_model": "kimi_code/k3",
        "producer_route_family": "codex_kimi_chat_completions_adapter",
    }


def test_build_route_identity_rejects_provider_alias_model():
    assert (
        erp.build_route_identity_from_provenance(
            {
                "producer_provider": "kimi_code",
                "producer_model": "provider-kimi_code",
                "producer_route_family": "codex_kimi_chat_completions_adapter",
            }
        )
        is None
    )


@pytest.mark.parametrize(
    "producer_model",
    ("None", "null", "model_id:None", "litellm_enc:model_id:None"),
)
def test_build_route_identity_rejects_none_and_model_id_none(producer_model):
    assert (
        erp.build_route_identity_from_provenance(
            {
                "producer_provider": "kimi_code",
                "producer_model": producer_model,
                "producer_route_family": "codex_kimi_chat_completions_adapter",
            }
        )
        is None
    )


def test_build_route_identity_rejects_missing_field():
    assert (
        erp.build_route_identity_from_provenance(
            {
                "producer_provider": "kimi_code",
                "producer_model": "kimi_code/k3",
            }
        )
        is None
    )


def test_build_producer_provenance_retains_actual_egress_over_conflicting_metadata():
    provenance = erp.build_producer_provenance_from_egress_context(
        custom_llm_provider="openai",
        expected_target_family="openai",
        egress_credential_family="codex_oauth",
        route_family="codex_oauth",
        request_body={
            "model": "gpt-5.6-sol",
            "litellm_metadata": {
                "codex_auto_agent_selected_provider": "xai",
                "codex_auto_agent_selected_model": "grok-4",
                "codex_auto_agent_selected_route_family": (
                    "codex_xai_oauth_responses_adapter"
                ),
            },
        },
    )
    assert provenance["producer_provider"] == "openai"
    assert provenance["producer_model"] == "gpt-5.6-sol"
    assert provenance["producer_route_family"] == "codex_oauth"
