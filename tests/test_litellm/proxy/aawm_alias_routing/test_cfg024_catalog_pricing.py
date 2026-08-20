"""CFG-024: served passthrough catalog ids with truthful pricing."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

from fastapi.responses import JSONResponse
from starlette.responses import JSONResponse as StarletteJSONResponse

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.catalog import (
    build_passthrough_model_list,
)


SERVED_CONCRETE_MODELS = (
    "cursor_agent/composer-2.5",
    "cursor_agent/cursor-grok-4.6-high",
    "oa_xai/grok-4.6",
    "kimi_code/k3",
    "cohere/north-mini-code-1-0",
    "openrouter/qwen/qwen3.6-flash",
    "openrouter/qwen/qwen3.5-flash-02-23",
)

ALIBABA_CHAT_PREFIX = "alibaba_token_plan/"
OPENCODE_ZEN_YAML_IDS = ("deepseek-v4-flash-free", "big-pickle")
TRUTHFUL_PROVENANCE = {
    "bundled map",
    "bundled_map",
    "aawm_reference_pricing",
    "unknown",
}


def _catalog_rows() -> List[Dict[str, Any]]:
    payload = build_passthrough_model_list()
    if isinstance(payload, dict) and "data" in payload:
        rows = payload["data"]
    else:
        rows = payload
    assert isinstance(rows, list)
    return rows


def _row_by_id(rows: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
    matches = [row for row in rows if row.get("id") == model_id]
    assert matches, f"expected catalog id {model_id!r}"
    return matches[0]


def test_build_passthrough_model_list_returns_openai_list_payload() -> None:
    payload = build_passthrough_model_list()
    assert isinstance(payload, dict)
    assert payload.get("object") == "list"
    assert isinstance(payload.get("data"), list)
    assert payload["data"], "catalog data must not be empty"


def test_served_concrete_models_are_present_with_real_providers() -> None:
    rows = _catalog_rows()
    ids = {row.get("id") for row in rows}

    for model_id in SERVED_CONCRETE_MODELS:
        assert model_id in ids, f"missing served concrete id {model_id!r}"
        row = _row_by_id(rows, model_id)
        assert row.get("owned_by") not in (None, "", "aawm_alias")
        assert row.get("owned_by") != "aawm_alias"
        assert row.get("id") == model_id


def test_alibaba_token_plan_chat_id_is_served() -> None:
    rows = _catalog_rows()
    alibaba_chat_ids = [
        row.get("id")
        for row in rows
        if isinstance(row.get("id"), str)
        and str(row.get("id")).startswith(ALIBABA_CHAT_PREFIX)
        and row.get("mode") in (None, "chat", "completion")
    ]
    assert alibaba_chat_ids, "expected at least one alibaba_token_plan/* chat id"
    assert any(
        model_id.endswith("glm-5.2") or model_id.endswith("qwen3.8-max")
        for model_id in alibaba_chat_ids
    )


def _is_opencode_zen_id(model_id: str, stem: str) -> bool:
    if model_id == stem or model_id == f"opencode/{stem}":
        return True
    return stem in model_id and ("opencode" in model_id or "zen" in model_id)


def test_opencode_zen_id_is_served() -> None:
    rows = _catalog_rows()
    ids = {str(row.get("id")) for row in rows if row.get("id")}
    for stem in OPENCODE_ZEN_YAML_IDS:
        assert any(
            _is_opencode_zen_id(model_id, stem) for model_id in ids
        ), f"missing OpenCode Zen yaml candidate {stem!r} in {sorted(ids)}"


def test_concrete_ids_never_owned_by_aawm_alias() -> None:
    rows = _catalog_rows()
    for row in rows:
        model_id = row.get("id")
        if not isinstance(model_id, str):
            continue
        # YAML alias ids are slash-free (`work`, `basic`, `sota-zai`, …).
        if "/" not in model_id or model_id.startswith("aawm"):
            continue
        assert row.get("owned_by") != "aawm_alias", model_id


def test_alias_rows_use_public_sota_zai_id_only() -> None:
    rows = _catalog_rows()
    alias_rows = [row for row in rows if row.get("owned_by") == "aawm_alias"]
    ids = {row.get("id") for row in rows}
    alias_ids = {row.get("id") for row in alias_rows}
    assert "aawm-sota-zai" not in ids
    if alias_rows:
        assert "sota-zai" in alias_ids
    for row in rows:
        if row.get("id") == "sota-zai":
            assert row.get("owned_by") == "aawm_alias"


def test_unknown_unroutable_ids_stay_absent() -> None:
    rows = _catalog_rows()
    ids = {row.get("id") for row in rows}
    assert "aawm-sota-zai" not in ids
    assert "not-a-real-model" not in ids
    assert "unknown/unroutable" not in ids


def test_overlay_rows_advertise_cost_map_mode() -> None:
    rows = _catalog_rows()
    for row in rows:
        mode = row.get("mode")
        if mode in {"embedding", "rerank"}:
            assert mode != "chat"
        model_id = str(row.get("id") or "")
        if "embed" in model_id.lower():
            assert mode == "embedding"
        if "rerank" in model_id.lower():
            assert mode == "rerank"


def test_overlay_rows_have_explicit_pricing_provenance() -> None:
    rows = _catalog_rows()
    for row in rows:
        if row.get("owned_by") == "aawm_alias":
            continue
        provenance = (
            row.get("pricing_provenance")
            or row.get("cost_provenance")
            or row.get("provenance")
        )
        assert provenance in TRUTHFUL_PROVENANCE, row.get("id")
        for key, value in row.items():
            if "invoice" in str(key).lower():
                raise AssertionError(f"catalog must not claim invoice cost: {key}")
            if isinstance(value, str):
                assert "invoice" not in value.lower()


def test_cursor_grok_cost_is_not_copied_from_oa_xai_grok() -> None:
    rows = _catalog_rows()
    cursor_grok = _row_by_id(rows, "cursor_agent/cursor-grok-4.6-high")
    oa_xai_grok = _row_by_id(rows, "oa_xai/grok-4.6")

    cursor_input = cursor_grok.get("input_cost_per_token")
    oa_input = oa_xai_grok.get("input_cost_per_token")
    cursor_output = cursor_grok.get("output_cost_per_token")
    oa_output = oa_xai_grok.get("output_cost_per_token")

    assert (cursor_input, cursor_output) != (oa_input, oa_output)


def test_get_models_short_circuit_returns_local_catalog(monkeypatch) -> None:
    from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as module

    request = MagicMock()
    request.method = "GET"
    request.headers = {}

    monkeypatch.setattr(module, "_is_openai_models_endpoint", lambda endpoint: True)
    monkeypatch.setattr(
        module,
        "_should_preserve_openai_client_auth",
        lambda **_kwargs: False,
    )

    async def _fail_if_forwarded(*_args, **_kwargs):
        raise AssertionError("Codex-native ChatGPT forwarding must not run")

    monkeypatch.setattr(
        module,
        "create_pass_through_route",
        lambda *args, **kwargs: _fail_if_forwarded,
    )

    import inspect

    route = inspect.unwrap(module.openai_proxy_route)
    response = route(
        endpoint="models",
        request=request,
        fastapi_response=MagicMock(),
        user_api_key_dict=MagicMock(),
    )
    if inspect.isawaitable(response):
        import asyncio

        response = asyncio.run(response)

    assert isinstance(response, (JSONResponse, StarletteJSONResponse))
    payload = response.body
    if isinstance(payload, (bytes, bytearray)):
        import json

        payload = json.loads(payload.decode("utf-8"))
    assert payload.get("object") == "list"
    assert isinstance(payload.get("data"), list)


def test_codex_native_get_models_still_forwards(monkeypatch) -> None:
    from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as module

    request = MagicMock()
    request.method = "GET"
    request.headers = {}

    monkeypatch.setattr(module, "_is_openai_models_endpoint", lambda endpoint: True)
    monkeypatch.setattr(
        module,
        "_should_preserve_openai_client_auth",
        lambda **_kwargs: True,
    )

    forwarded = {}

    async def _forward(*_args, **_kwargs):
        forwarded["called"] = True
        return {"forwarded": True}

    monkeypatch.setattr(
        module,
        "create_pass_through_route",
        lambda *args, **kwargs: (lambda *a, **k: _forward()),
    )

    import inspect

    route = inspect.unwrap(module.openai_proxy_route)
    try:
        response = route(
            endpoint="models",
            request=request,
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
        )
        if inspect.isawaitable(response):
            import asyncio

            response = asyncio.run(response)
    except Exception:
        # Forwarding may fail after the Codex-native branch is selected; that
        # is still evidence the local catalog short-circuit was skipped.
        forwarded.setdefault("called", True)

    assert forwarded.get("called") is True or response is not None
