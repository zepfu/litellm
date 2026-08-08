from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pytest

from litellm.proxy.pass_through_endpoints import (
    aawm_claude_control_plane as control_plane,
)
from litellm.proxy.pass_through_endpoints import aawm_context_query as context_query
from litellm.proxy.pass_through_endpoints.aawm_request_policy import (
    claude_prompt_replacement,
)


@pytest.mark.asyncio
async def test_context_query_cache_uses_injected_ttl_and_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"value": 100.0}
    secrets = {"AAWM_DYNAMIC_INJECTION_CACHE_TTL_SECONDS": "5"}
    monkeypatch.setattr(
        context_query,
        "_runtime",
        context_query.ContextQueryRuntime(
            get_secret_str=lambda name: secrets.get(name),
            monotonic=lambda: now["value"],
        ),
    )
    context_query._aawm_dynamic_injection_cache.clear()
    context_query._aawm_context_grab_cache.clear()

    dynamic_key = ("proc", "session", "agent", "tenant")
    context_key = ("proc", "session", "tenant", "agent", "name")
    await context_query._set_cached_aawm_dynamic_injection_result(
        dynamic_key,
        "memory",
    )
    await context_query._set_cached_aawm_context_grab_result(
        context_key,
        {"content": "context"},
    )

    assert await context_query._get_cached_aawm_dynamic_injection_result(dynamic_key) == (True, "memory")
    assert await context_query._get_cached_aawm_context_grab_result(context_key) == (
        True,
        {"content": "context"},
    )

    now["value"] = 106.0
    assert await context_query._get_cached_aawm_dynamic_injection_result(dynamic_key) == (False, None)
    assert await context_query._get_cached_aawm_context_grab_result(context_key) == (
        False,
        None,
    )


@pytest.mark.asyncio
async def test_context_query_preserves_config_and_query_failure_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        context_query,
        "_runtime",
        context_query.ContextQueryRuntime(get_secret_str=lambda _name: None),
    )
    monkeypatch.setattr(context_query, "_aawm_dynamic_injection_pool", None)

    with pytest.raises(
        RuntimeError,
        match="AAWM dynamic injection database configuration is missing",
    ):
        await context_query._get_aawm_dynamic_injection_pool()

    acquire_timeouts: list[float] = []

    class _Connection:
        async def fetch(self, _query: str, *_args: Any) -> Any:
            raise LookupError("query failed")

    class _Acquire:
        async def __aenter__(self) -> _Connection:
            return _Connection()

        async def __aexit__(self, *_exc: Any) -> bool:
            return False

    class _Pool:
        def acquire(self, *, timeout: float) -> _Acquire:
            acquire_timeouts.append(timeout)
            return _Acquire()

    with pytest.raises(LookupError, match="query failed"):
        await context_query._aawm_pool_fetch(
            _Pool(),
            "SELECT content",
            "name",
            get_timeout=lambda: 1.25,
        )
    assert acquire_timeouts == [1.25]


@pytest.mark.asyncio
async def test_context_query_pool_preserves_size_timeout_and_statement_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_pool_calls: list[dict[str, Any]] = []
    created_pool = object()

    class _Asyncpg:
        async def create_pool(self, **kwargs: Any) -> Any:
            create_pool_calls.append(kwargs)
            return created_pool

    monkeypatch.setattr(
        context_query,
        "_runtime",
        context_query.ContextQueryRuntime(
            get_secret_str=lambda _name: None,
            import_module=lambda name: (_Asyncpg() if name == "asyncpg" else pytest.fail(f"unexpected import: {name}")),
        ),
    )
    monkeypatch.setattr(context_query, "_aawm_dynamic_injection_pool", None)

    async def initialize_connection(_connection: Any) -> None:
        return None

    pool = await context_query._get_aawm_dynamic_injection_pool(
        build_dsn=lambda: "postgresql://aawm@db/aawm",
        get_server_settings=lambda: {"application_name": "aawm-test"},
        initialize_connection=initialize_connection,
    )

    assert pool is created_pool
    assert create_pool_calls == [
        {
            "dsn": "postgresql://aawm@db/aawm",
            "min_size": 1,
            "max_size": 4,
            "command_timeout": 10,
            "statement_cache_size": 0,
            "server_settings": {"application_name": "aawm-test"},
            "init": initialize_connection,
        }
    ]


@pytest.mark.asyncio
async def test_rewriter_uses_explicit_services_and_deterministic_timestamps() -> None:
    cache: dict[context_query.DynamicCacheKey, Optional[str]] = {}
    memory_calls: list[tuple[str, str]] = []
    timestamps = iter(
        (
            datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 8, 12, 0, 1, tzinfo=timezone.utc),
        )
    )

    async def get_cached(
        key: context_query.DynamicCacheKey,
    ) -> context_query.DynamicCacheResult:
        return key in cache, cache.get(key)

    async def set_cached(
        key: context_query.DynamicCacheKey,
        value: Optional[str],
    ) -> None:
        cache[key] = value

    async def get_context_cached(
        _key: context_query.ContextCacheKey,
    ) -> context_query.ContextCacheResult:
        return False, None

    async def set_context_cached(
        _key: context_query.ContextCacheKey,
        _value: dict[str, str],
    ) -> None:
        return None

    async def get_agent_memories(
        *,
        agent_name: str,
        tenant_id: str,
    ) -> Optional[str]:
        memory_calls.append((agent_name, tenant_id))
        return "# Memory Injection\nremembered"

    async def no_context(**_kwargs: Any) -> Optional[str]:
        return None

    def merge_metadata(
        body: dict[str, Any],
        *,
        tags_to_add: list[str],
        extra_fields: dict[str, Any],
    ) -> dict[str, Any]:
        updated = dict(body)
        metadata = dict(updated.get("litellm_metadata") or {})
        metadata["tags"] = tags_to_add
        metadata.update(extra_fields)
        updated["litellm_metadata"] = metadata
        return updated

    services = control_plane.build_claude_control_plane_services(
        prompt=claude_prompt_replacement.build_claude_prompt_replacement_services(),
        context_query=context_query.ContextQueryServices(
            get_cached_dynamic_result=get_cached,
            set_cached_dynamic_result=set_cached,
            get_cached_context_result=get_context_cached,
            set_cached_context_result=set_context_cached,
            get_agent_memories=get_agent_memories,
            get_context=no_context,
            get_reference_identifiers=no_context,
            get_context_proc_name=lambda: "tristore_search_exact",
            get_context_proc_name_for_logging=lambda: "tristore_search_exact",
            max_parallel_queries=4,
        ),
        now_utc=lambda: next(timestamps),
        merge_metadata=merge_metadata,
        build_span=lambda **kwargs: dict(kwargs),
        format_span_timestamp=lambda value: value.isoformat(),
        add_context_file_metadata=lambda body: body,
    )
    rewriter = control_plane.compose_claude_control_plane(services)
    request_body = {
        "system": (
            "You are 'writer' and you are working on the 'litellm' project\n"
            "@@@ AAWM p=get_agent_memories ctx=agent,tenant @@@"
        ),
        "messages": [{"role": "user", "content": "continue"}],
    }

    updated, events = await rewriter.expand_dynamic_context(request_body)

    assert memory_calls == [("writer", "litellm")]
    assert "remembered" in updated["system"]
    assert "@@@ AAWM" not in updated["system"]
    assert events[0]["status"] == "resolved"
    assert events[0]["cache_status"] == "miss"
    span = updated["litellm_metadata"]["langfuse_spans"][0]
    assert span["name"] == "aawm.dynamic_injection"
    assert span["start_time"] == "2026-08-08T12:00:00+00:00"
    assert span["end_time"] == "2026-08-08T12:00:01+00:00"


def test_control_plane_has_only_thin_prompt_and_context_compatibility_ownership() -> None:
    control_plane_path = Path(control_plane.__file__).resolve()
    context_query_path = Path(context_query.__file__).resolve()
    prompt_path = Path(claude_prompt_replacement.__file__).resolve()
    control_plane_source = control_plane_path.read_text(encoding="utf-8")
    control_plane_tree = ast.parse(control_plane_source)
    context_query_tree = ast.parse(context_query_path.read_text(encoding="utf-8"))
    prompt_tree = ast.parse(prompt_path.read_text(encoding="utf-8"))

    control_plane_defs = {
        node.name for node in control_plane_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    prompt_defs = {node.name for node in prompt_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    canonical_prompt_defs = {
        "_candidate_context_replacement_dirs",
        "_resolve_context_replacement_file",
        "_parse_claude_code_version",
        "_resolve_claude_auto_memory_template_path",
        "_load_claude_context_replacement_template",
        "_resolve_claude_prompt_patch_manifest_path",
        "_load_claude_prompt_patch_manifest",
        "_render_claude_auto_memory_replacement",
        "_replace_claude_auto_memory_section_in_text",
        "_apply_claude_prompt_patch_manifest_to_text",
    }

    assert canonical_prompt_defs <= prompt_defs
    assert canonical_prompt_defs.isdisjoint(control_plane_defs)
    assert "llm_passthrough_endpoints" not in control_plane_source
    assert "def _lp(" not in control_plane_source
    assert "asyncpg" not in control_plane_source
    assert "_AAWM_REFERENCE_IDENTIFIER_LIST_QUERY" not in control_plane_source

    pool_owners = [
        node
        for node in ast.walk(context_query_tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_aawm_dynamic_injection_pool"
    ]
    assert len(pool_owners) == 1
    assert not any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            any(
                isinstance(target, ast.Name) and target.id == "_aawm_dynamic_injection_pool"
                for target in getattr(node, "targets", ())
            )
            or (
                isinstance(getattr(node, "target", None), ast.Name) and node.target.id == "_aawm_dynamic_injection_pool"
            )
        )
        for node in ast.walk(control_plane_tree)
    )
