"""CFG-023: publish compiled YAML alias names on the Codex passthrough /models list."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _alias_entry(
    *,
    name: str,
    target: str = "openai/gpt-5.4",
    provider: str = "openai",
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        target=target,
        provider=provider,
        owned_by="aawm_alias",
    )


def _snapshot_with_aliases(names: List[str]) -> SimpleNamespace:
    aliases = {_name: _alias_entry(name=_name) for _name in names}
    return SimpleNamespace(aliases=aliases)


COMPILED_ALIAS_NAMES = [
    "work",
    "work-other",
    "sota-zai",
    "investigate",
    "plan",
]


UNKNOWN_ALIAS_NAME = "totally-unknown-alias-xyz"
AAWM_PREFIXED_SOTA = "aawm-sota-zai"


class TestPassthroughCatalogHelpers:
    def test_iter_compiled_alias_names_from_snapshot(self) -> None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.catalog import (
            iter_compiled_alias_names,
        )

        snapshot = _snapshot_with_aliases(COMPILED_ALIAS_NAMES)
        names = list(iter_compiled_alias_names(snapshot))
        for expected in COMPILED_ALIAS_NAMES:
            assert expected in names
        assert AAWM_PREFIXED_SOTA not in names
        assert UNKNOWN_ALIAS_NAME not in names

    def test_iter_compiled_alias_names_fail_closed_on_none(self) -> None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.catalog import (
            iter_compiled_alias_names,
        )

        assert list(iter_compiled_alias_names(None)) == []

    def test_build_passthrough_model_list_contains_yaml_names(self) -> None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.catalog import (
            build_passthrough_model_list,
        )

        snapshot = _snapshot_with_aliases(COMPILED_ALIAS_NAMES)
        payload = build_passthrough_model_list(snapshot)
        assert payload["object"] == "list"
        ids = [row["id"] for row in payload["data"]]
        for expected in COMPILED_ALIAS_NAMES:
            assert expected in ids
        assert "sota-zai" in ids
        assert "work-other" in ids
        assert AAWM_PREFIXED_SOTA not in ids
        assert UNKNOWN_ALIAS_NAME not in ids
        for row in payload["data"]:
            assert row["object"] == "model"
            assert row["owned_by"] == "aawm_alias"

    def test_build_passthrough_model_list_fail_closed_on_none(self) -> None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.catalog import (
            build_passthrough_model_list,
        )

        payload = build_passthrough_model_list(None)
        assert payload == {"object": "list", "data": []}

    def test_swapped_snapshot_add_remove_appears_on_list(self) -> None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.catalog import (
            build_passthrough_model_list,
        )

        first = _snapshot_with_aliases(["work", "sota-zai"])
        first_ids = {row["id"] for row in build_passthrough_model_list(first)["data"]}
        assert first_ids == {"work", "sota-zai"}

        second = _snapshot_with_aliases(["work", "work-other"])
        second_ids = {row["id"] for row in build_passthrough_model_list(second)["data"]}
        assert "work" in second_ids
        assert "work-other" in second_ids
        assert "sota-zai" not in second_ids


def _openai_models_request(*, endpoint: str = "v1/models") -> MagicMock:
    request = MagicMock()
    request.method = "GET"
    request.url.path = f"/openai_passthrough/{endpoint}"
    request.headers = {}
    request.query_params = {}
    request.state = SimpleNamespace()
    return request


@pytest.mark.asyncio
async def test_non_codex_native_models_list_contains_compiled_aliases() -> None:
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        openai_proxy_route,
    )

    snapshot = _snapshot_with_aliases(COMPILED_ALIAS_NAMES)
    request = _openai_models_request()

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._is_openai_models_endpoint",
        return_value=True,
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._should_preserve_openai_client_auth",
        return_value=False,
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._request_uses_codex_native_auth",
        return_value=False,
    ), patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.catalog.get_active_snapshot",
        return_value=snapshot,
    ):
        response = await openai_proxy_route(
            endpoint="v1/models",
            request=request,
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
        )

    payload = response if isinstance(response, dict) else getattr(response, "body", response)
    if hasattr(payload, "body"):
        import json

        payload = json.loads(payload.body.decode("utf-8"))
    elif not isinstance(payload, dict) and hasattr(payload, "json"):
        maybe = payload.json()
        payload = maybe if isinstance(maybe, dict) else payload

    # Starlette JSONResponse stores content on `.body` as bytes.
    if not isinstance(payload, dict):
        import json

        from starlette.responses import JSONResponse

        assert isinstance(response, JSONResponse)
        payload = json.loads(response.body.decode("utf-8"))

    ids = [row["id"] for row in payload["data"]]
    for expected in COMPILED_ALIAS_NAMES:
        assert expected in ids
    assert "sota-zai" in ids
    assert "work-other" in ids
    assert AAWM_PREFIXED_SOTA not in ids
    assert UNKNOWN_ALIAS_NAME not in ids


@pytest.mark.asyncio
async def test_swapped_snapshot_appears_on_non_codex_models_list() -> None:
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        openai_proxy_route,
    )

    request = _openai_models_request()

    async def _list_ids(snapshot: SimpleNamespace) -> List[str]:
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._is_openai_models_endpoint",
            return_value=True,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._should_preserve_openai_client_auth",
            return_value=False,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._request_uses_codex_native_auth",
            return_value=False,
        ), patch(
            "litellm.proxy.pass_through_endpoints.aawm_alias_routing.catalog.get_active_snapshot",
            return_value=snapshot,
        ):
            response = await openai_proxy_route(
                endpoint="v1/models",
                request=request,
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
            )
        import json

        from starlette.responses import JSONResponse

        assert isinstance(response, JSONResponse)
        payload = json.loads(response.body.decode("utf-8"))
        return [row["id"] for row in payload["data"]]

    first_ids = await _list_ids(_snapshot_with_aliases(["work", "sota-zai"]))
    assert "work" in first_ids
    assert "sota-zai" in first_ids
    assert "work-other" not in first_ids

    second_ids = await _list_ids(_snapshot_with_aliases(["work", "work-other"]))
    assert "work" in second_ids
    assert "work-other" in second_ids
    assert "sota-zai" not in second_ids


@pytest.mark.asyncio
async def test_codex_native_models_still_targets_chatgpt_codex_models() -> None:
    """Codex-native GET /models keeps ChatGPT Codex forwarding and does not take a session-owner lock."""
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        openai_proxy_route,
    )

    request = _openai_models_request()
    captured: Dict[str, Any] = {}

    async def _fake_passthrough(*args: Any, **kwargs: Any) -> Dict[str, str]:
        captured["args"] = args
        captured.update(kwargs)
        return {"object": "list", "data": []}

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._is_openai_models_endpoint",
        return_value=True,
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._should_preserve_openai_client_auth",
        return_value=True,
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._request_uses_codex_native_auth",
        return_value=True,
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
        return_value=_fake_passthrough,
    ) as create_route, patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.catalog.build_passthrough_model_list",
    ) as build_list:
        result = await openai_proxy_route(
            endpoint="v1/models",
            request=request,
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
        )

    assert result == {"object": "list", "data": []}
    route_kwargs = create_route.call_args.kwargs if create_route.call_args else {}
    target = str(route_kwargs.get("target") or captured.get("url") or captured.get("target") or "")
    assert "chatgpt.com/backend-api/codex/models" in target
    build_list.assert_not_called()
    # Session-owner lock must not be taken on model-list paths.
    assert "session_owner" not in captured
    assert captured.get("take_session_owner_lock") in (None, False)


@pytest.mark.asyncio
async def test_post_passthrough_responses_model_work_still_resolves_alias() -> None:
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        openai_proxy_route,
    )

    request = MagicMock()
    request.method = "POST"
    request.url.path = "/openai_passthrough/v1/responses"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}
    request.state = SimpleNamespace()
    dispatch = AsyncMock(return_value={"id": "alias-ok", "model": "work"})
    handler = AsyncMock(return_value={"id": "alias-ok", "model": "work"})

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._is_openai_models_endpoint",
        return_value=False,
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
        AsyncMock(return_value={"model": "work", "input": "hello"}),
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._resolve_codex_auto_agent_alias_model",
        return_value="work",
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._is_openai_responses_endpoint",
        return_value=True,
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.BaseOpenAIPassThroughHandler._base_openai_pass_through_handler",
        handler,
    ):
        import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints as lpe

        lpe.try_dispatch_codex_request = dispatch
        result = await openai_proxy_route(
            endpoint="v1/responses",
            request=request,
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
        )

    assert result == {"id": "alias-ok", "model": "work"}
    assert handler.await_count >= 1 or dispatch.await_count >= 1


def test_proxy_server_chat_completion_unchanged() -> None:
    import litellm.proxy.proxy_server as proxy_server

    source = inspect.getsource(proxy_server.chat_completion)
    assert "build_passthrough_model_list" not in source
    assert "iter_compiled_alias_names" not in source
    assert "aawm_alias_routing.catalog" not in source


def test_no_new_model_list_row_named_after_an_alias() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    # Walk likely yaml configs; compiled aliases must not be added as model_list rows.
    yaml_candidates = [
        repo_root / "litellm-dev-config.yaml",
        repo_root / "proxy" / "example_config_yaml",
    ]
    alias_ids = {"sota-zai", "work-other", "aawm-sota-zai"}
    hits: List[str] = []
    for candidate in yaml_candidates:
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8")
            if "model_list" not in text:
                continue
            for alias in alias_ids:
                # A model_name / model_list row named after the alias is forbidden.
                if f"model_name: {alias}" in text or f"model_name: '{alias}'" in text:
                    hits.append(f"{candidate}: {alias}")
        elif candidate.is_dir():
            for yaml_file in candidate.rglob("*.yaml"):
                text = yaml_file.read_text(encoding="utf-8")
                if "model_list" not in text:
                    continue
                for alias in alias_ids:
                    if f"model_name: {alias}" in text or f"model_name: '{alias}'" in text:
                        hits.append(f"{yaml_file}: {alias}")
    assert hits == []
