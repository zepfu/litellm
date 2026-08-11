"""OPENAI-006 encrypted-reasoning provenance and OpenAI egress compatibility."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    encrypted_reasoning_provenance as erp,
)


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
    meta = body["litellm_metadata"]
    assert meta["encrypted_reasoning_disposition"] == "allowed_compatible"
    assert getattr(request.state, "_aawm_encrypted_reasoning_disposition", None)


def test_streaming_sse_stamp_preserves_ciphertext():
    from litellm.proxy.pass_through_endpoints.streaming_handler import (
        PassThroughStreamingHandler,
    )

    event = {
        "type": "response.output_item.done",
        "item": {
            "type": "reasoning",
            "id": "rs_stream_1",
            "encrypted_content": CIPHERTEXT,
        },
    }
    import json

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
    mock_httpx_response.aiter_bytes = AsyncMock(
        return_value=iter([b'data: {"type":"response.completed"}\n\n'])
    )
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
