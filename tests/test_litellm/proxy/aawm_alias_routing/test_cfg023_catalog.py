"""CFG-023: publish compiled YAML alias names on the Codex passthrough /models list."""

from __future__ import annotations

import inspect
import logging
from types import SimpleNamespace
from typing import Any, Dict, List
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


def test_alias_catalog_publishes_configured_multi_agent_version_only() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.catalog import (
        build_passthrough_model_list,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        DEFAULT_CONFIG_DIR,
        compile_directory,
    )

    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    rows = build_passthrough_model_list(snapshot)["data"]
    rows_by_id = {row["id"]: row for row in rows}

    assert rows_by_id["sota-xai"]["multi_agent_version"] == "v2"
    assert "multi_agent_version" not in rows_by_id["work"]

    without_metadata = SimpleNamespace(
        aliases={name: SimpleNamespace() for name in snapshot.aliases}
    )
    baseline_rows = build_passthrough_model_list(without_metadata)["data"]
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    assert rows_by_id["oa_xai/grok-4.6"] == baseline_by_id["oa_xai/grok-4.6"]


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
        yaml_ids = set(COMPILED_ALIAS_NAMES)
        for row in payload["data"]:
            assert row["object"] == "model"
            if row.get("owned_by") == "aawm_alias":
                assert row["id"] in yaml_ids
            elif row["id"] in yaml_ids:
                assert row["owned_by"] == "aawm_alias"

    def test_build_passthrough_model_list_fail_closed_on_none(self) -> None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.catalog import (
            build_passthrough_model_list,
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_alias_routing.catalog.get_active_snapshot",
            return_value=None,
        ):
            payload = build_passthrough_model_list(None)
        assert payload["object"] == "list"
        assert isinstance(payload["data"], list)
        alias_ids = [
            row["id"] for row in payload["data"] if row.get("owned_by") == "aawm_alias"
        ]
        ids = {row["id"] for row in payload["data"]}
        assert alias_ids == []
        assert "work" not in ids
        assert "sota-zai" not in ids

    def test_swapped_snapshot_add_remove_appears_on_list(self) -> None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.catalog import (
            build_passthrough_model_list,
        )

        first = _snapshot_with_aliases(["work", "sota-zai"])
        first_ids = {row["id"] for row in build_passthrough_model_list(first)["data"]}
        assert "work" in first_ids
        assert "sota-zai" in first_ids

        second = _snapshot_with_aliases(["work", "work-other"])
        second_ids = {row["id"] for row in build_passthrough_model_list(second)["data"]}
        assert "work" in second_ids
        assert "work-other" in second_ids
        assert "sota-zai" not in second_ids


def _openai_models_request(*, endpoint: str = "v1/models") -> MagicMock:
    path = f"/openai_passthrough/{endpoint}"
    request = MagicMock()
    request.method = "GET"
    request.url.path = path
    request.headers = {}
    request.query_params = {}
    request.state = SimpleNamespace()
    request.scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "query_string": b"",
        "client": ("172.18.0.1", 50324),
        "http_version": "1.1",
    }
    return request


def _uvicorn_access_record(*, method: str, full_path: str, status_code: int = 200) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("172.18.0.1:50324", method, full_path, "1.1", status_code),
        exc_info=None,
    )


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
@pytest.mark.parametrize(
    "endpoint,path",
    [
        ("v1/models", "/openai_passthrough/v1/models"),
        ("models", "/openai_passthrough/models"),
    ],
)
async def test_non_codex_native_models_list_replaces_uvicorn_access_log(
    endpoint: str,
    path: str,
) -> None:
    """CFG-023 local catalog GET must consume the matching uvicorn access line."""
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        openai_proxy_route,
    )
    from litellm._logging import (
        AawmRouteAccessLogReplacementFilter,
        clear_aawm_route_access_log_replacements,
    )

    clear_aawm_route_access_log_replacements()
    snapshot = _snapshot_with_aliases(COMPILED_ALIAS_NAMES)
    request = _openai_models_request(endpoint=endpoint)

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
            endpoint=endpoint,
            request=request,
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
        )

    from starlette.responses import JSONResponse

    assert isinstance(response, JSONResponse)
    access_filter = AawmRouteAccessLogReplacementFilter()
    matching = _uvicorn_access_record(method="GET", full_path=path)
    assert access_filter.filter(matching) is False
    assert access_filter.filter(matching) is True
    clear_aawm_route_access_log_replacements()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint,path,status_code",
    [
        ("model_group/info", "/openai_passthrough/model_group/info", 404),
        ("model/info", "/openai_passthrough/model/info", 404),
        ("v1/model/info", "/openai_passthrough/v1/model/info", 404),
        ("v2/model/info", "/openai_passthrough/v2/model/info", 404),
    ],
)
async def test_openai_passthrough_model_info_probes_replace_uvicorn_access_log(
    endpoint: str,
    path: str,
    status_code: int,
) -> None:
    """Ohmypi catalog discovery GETs under /openai_passthrough must consume leftover uvicorn."""
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        openai_proxy_route,
    )
    from litellm._logging import (
        AawmRouteAccessLogReplacementFilter,
        clear_aawm_route_access_log_replacements,
    )

    clear_aawm_route_access_log_replacements()
    request = _openai_models_request(endpoint=endpoint)
    handler = AsyncMock(return_value={"object": "list", "data": []})

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._should_preserve_openai_client_auth",
        return_value=False,
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
        return_value="sk-test",
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.BaseOpenAIPassThroughHandler._base_openai_pass_through_handler",
        handler,
    ):
        await openai_proxy_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
        )

    access_filter = AawmRouteAccessLogReplacementFilter()
    matching = _uvicorn_access_record(
        method="GET",
        full_path=path,
        status_code=status_code,
    )
    assert access_filter.filter(matching) is False
    assert access_filter.filter(matching) is True
    clear_aawm_route_access_log_replacements()


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
