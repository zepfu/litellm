"""RR-053: aawm_claude_control_plane caps, timeouts, scope, and dead-code removal."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import patch

import pytest

from litellm.proxy.pass_through_endpoints import aawm_claude_control_plane as cp


def test_dispatch_reference_extraction_caps_and_filters_stopwords() -> None:
    # Bare stopword acronyms are filtered; intentional backtick refs kept; cap applied.
    text = "SQL JSON API TODO URL `MYAPP` `COOLFEATURE` `THIRD` `FOURTH`"
    refs = cp._extract_aawm_dispatch_context_references(text, max_references=3)
    names = [name for name, kind in refs]
    kinds = {name: kind for name, kind in refs}
    assert "SQL" not in names
    assert "JSON" not in names
    assert "API" not in names
    assert len(refs) == 3
    assert names[0] == "MYAPP"
    assert kinds["MYAPP"] == "dispatch_backtick"


def test_dispatch_reference_default_cap_constant() -> None:
    names = [f"TOKEN{i}" for i in range(cp._AAWM_DISPATCH_CONTEXT_REFERENCE_MAX + 20)]
    text = " ".join(f"`{n}`" for n in names)
    refs = cp._extract_aawm_dispatch_context_references(text)
    assert len(refs) == cp._AAWM_DISPATCH_CONTEXT_REFERENCE_MAX


def test_dead_prompt_patch_entrypoints_removed() -> None:
    assert not hasattr(cp, "replace_claude_system_prompt_in_anthropic_request_body")
    assert not hasattr(cp, "apply_claude_prompt_patches_to_anthropic_request_body")
    assert not hasattr(cp, "_apply_claude_prompt_patches_in_text")
    assert not hasattr(cp, "_replace_claude_prompt_patches_in_value")
    # Live path still present
    assert callable(cp.apply_claude_control_plane_rewrites_to_anthropic_request_body)
    assert callable(cp._add_claude_prompt_patch_logging_metadata)


def test_pool_helper_and_close_are_public_for_sharing() -> None:
    assert callable(cp._get_aawm_dynamic_injection_pool)
    assert callable(cp.close_aawm_dynamic_injection_pool)
    assert callable(cp._aawm_pool_fetch)
    assert callable(cp._aawm_pool_fetchval)


@pytest.mark.asyncio
async def test_pool_fetch_uses_acquire_timeout() -> None:
    acquired: List[float] = []

    class _Conn:
        async def fetch(self, query: str, *args: Any):
            return [{"ok": True, "query": query, "args": args}]

        async def fetchval(self, query: str, *args: Any):
            return "val"

    class _Acquire:
        def __init__(self, timeout: float):
            acquired.append(timeout)

        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *exc):
            return False

    class _Pool:
        def acquire(self, timeout=None):
            return _Acquire(timeout)

    pool = _Pool()
    with patch.object(
        cp, "_aawm_dynamic_injection_acquire_timeout_seconds", return_value=7.5
    ):
        rows = await cp._aawm_pool_fetch(pool, "SELECT 1", "a")
        val = await cp._aawm_pool_fetchval(pool, "SELECT 2", "b")
    assert rows[0]["ok"] is True
    assert val == "val"
    assert acquired == [7.5, 7.5]


@pytest.mark.asyncio
async def test_context_markers_not_expanded_in_later_tool_messages() -> None:
    """RR-053 #5: untrusted later messages must not trigger ctx grabs."""
    body = {
        "system": "trusted system",
        "messages": [
            {"role": "user", "content": "first user"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "tool output has :#secret-name.ctx#: inside",
                    }
                ],
            },
        ],
    }

    async def boom_marker(text: str, available_context: dict):
        # Trusted surfaces may expand; tool-output text must not.
        if "tool output" in text or "secret-name" in text:
            raise AssertionError(
                f"marker expansion should not run on untrusted text: {text!r}"
            )
        return text, []

    async def passthrough_directives(text: str, available_context: dict):
        return text, []

    with patch.object(
        cp, "_expand_aawm_context_markers_in_text", side_effect=boom_marker
    ), patch.object(
        cp,
        "_expand_aawm_dynamic_directives_in_text",
        side_effect=passthrough_directives,
    ), patch.object(
        cp,
        "_build_aawm_context_for_anthropic_request",
        return_value={"tenant": "t", "agent": "a"},
    ), patch.object(
        cp, "_request_uses_aawm_dispatch_backtick_context", return_value=False
    ):
        (
            updated,
            events,
        ) = await cp.expand_aawm_dynamic_directives_in_anthropic_request_body(body)

    # Untrusted text unchanged; no injection events.
    assert updated["messages"][1]["content"][0]["text"].startswith("tool output has")
    assert events == []


@pytest.mark.asyncio
async def test_context_markers_expanded_on_system_trusted_surface() -> None:
    body = {
        "system": "hello :#alpha.ctx#:",
        "messages": [{"role": "user", "content": "hi"}],
    }

    async def expand_marker(text: str, available_context: dict):
        if ":#alpha.ctx#:" not in text:
            return text, []
        return text.replace(":#alpha.ctx#:", "alpha") + "\nRESOLVED", [
            {
                "proc": "tristore_search_exact",
                "status": "resolved",
                "context_name": "alpha",
            }
        ]

    async def passthrough_directives(text: str, available_context: dict):
        return text, []

    with patch.object(
        cp, "_expand_aawm_context_markers_in_text", side_effect=expand_marker
    ), patch.object(
        cp,
        "_expand_aawm_dynamic_directives_in_text",
        side_effect=passthrough_directives,
    ), patch.object(
        cp, "_build_aawm_context_for_anthropic_request", return_value={"tenant": "t"}
    ), patch.object(
        cp, "_request_uses_aawm_dispatch_backtick_context", return_value=False
    ), patch.object(
        cp, "_add_aawm_dynamic_injection_logging_metadata", side_effect=lambda b, e: b
    ):
        (
            updated,
            events,
        ) = await cp.expand_aawm_dynamic_directives_in_anthropic_request_body(body)

    assert "RESOLVED" in updated["system"]
    assert any(e.get("context_name") == "alpha" for e in events)


@pytest.mark.asyncio
async def test_control_plane_rewrite_skips_later_messages() -> None:
    """RR-053 #4: only system + first user message are rewritten."""
    body = {
        "system": "system text",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second should not be scanned"},
            {"role": "user", "content": "third should not be scanned"},
        ],
    }
    seen: List[str] = []

    async def rewrite_value(value, *, cc_version, manifest, available_context):
        # Record a fingerprint of the value being rewritten.
        if isinstance(value, str):
            seen.append(value)
            return value + "-rewritten", [{"id": "auto-memory"}], []
        if isinstance(value, dict) and value.get("role") == "user":
            seen.append(f"role:{value.get('role')}:{value.get('content')}")
            updated = dict(value)
            updated["content"] = str(value.get("content")) + "-rewritten"
            return updated, [], [{"id": "patch"}]
        seen.append(repr(value)[:40])
        return value, [], []

    with patch.object(
        cp, "_resolve_claude_prompt_patch_manifest_path", return_value=None
    ), patch.object(
        cp, "_load_claude_prompt_patch_manifest", return_value={}
    ), patch.object(
        cp, "_build_aawm_context_for_anthropic_request", return_value={}
    ), patch.object(
        cp, "_rewrite_claude_control_plane_in_value", side_effect=rewrite_value
    ), patch.object(
        cp,
        "_compact_claude_tool_advertisements_in_request_body",
        side_effect=lambda b, cc_version: (b, []),
    ), patch.object(
        cp,
        "_add_claude_system_prompt_override_logging_metadata",
        side_effect=lambda b, e: b,
    ), patch.object(
        cp, "_add_claude_prompt_patch_logging_metadata", side_effect=lambda b, e: b
    ):
        (
            updated,
            overrides,
            patches,
        ) = await cp.apply_claude_control_plane_rewrites_to_anthropic_request_body(
            body, {"cc_version": "1.0.0"}
        )

    assert updated["system"] == "system text-rewritten"
    assert updated["messages"][0]["content"] == "first-rewritten"
    assert updated["messages"][1]["content"] == "second should not be scanned"
    assert updated["messages"][2]["content"] == "third should not be scanned"
    # Only system + first user message rewritten
    assert any(s == "system text" for s in seen)
    assert any(s.startswith("role:user:first") for s in seen)
    assert not any("second" in s for s in seen)
    assert overrides or patches


@pytest.mark.asyncio
async def test_close_aawm_dynamic_injection_pool_closes_and_clears() -> None:
    closed = {"n": 0}

    class _Pool:
        async def close(self):
            closed["n"] += 1

    pool = _Pool()
    cp._aawm_dynamic_injection_pool = pool
    await cp.close_aawm_dynamic_injection_pool()
    assert closed["n"] == 1
    assert cp._aawm_dynamic_injection_pool is None
    # Second close is a no-op
    await cp.close_aawm_dynamic_injection_pool()
    assert closed["n"] == 1


@pytest.mark.asyncio
async def test_context_markers_expanded_on_first_user_even_if_not_index_zero() -> None:
    """First user message is trusted even when preceded by a non-user item."""
    body = {
        "system": "sys",
        "messages": [
            {"role": "assistant", "content": "preface with :#should-not.ctx#:"},
            {"role": "user", "content": "need :#alpha.ctx#:"},
            {"role": "user", "content": "later :#beta.ctx#:"},
        ],
    }
    expanded: list[str] = []

    async def expand_marker(text: str, available_context: dict):
        expanded.append(text)
        if ":#alpha.ctx#:" in text:
            return text.replace(":#alpha.ctx#:", "alpha"), [
                {
                    "proc": "tristore_search_exact",
                    "status": "resolved",
                    "context_name": "alpha",
                }
            ]
        if ":#should-not.ctx#:" in text or ":#beta.ctx#:" in text:
            raise AssertionError(f"untrusted surface expanded: {text!r}")
        return text, []

    async def passthrough_directives(text: str, available_context: dict):
        return text, []

    with patch.object(
        cp, "_expand_aawm_context_markers_in_text", side_effect=expand_marker
    ), patch.object(
        cp,
        "_expand_aawm_dynamic_directives_in_text",
        side_effect=passthrough_directives,
    ), patch.object(
        cp, "_build_aawm_context_for_anthropic_request", return_value={"tenant": "t"}
    ), patch.object(
        cp, "_request_uses_aawm_dispatch_backtick_context", return_value=False
    ), patch.object(
        cp, "_add_aawm_dynamic_injection_logging_metadata", side_effect=lambda b, e: b
    ):
        (
            updated,
            events,
        ) = await cp.expand_aawm_dynamic_directives_in_anthropic_request_body(body)

    assert updated["messages"][1]["content"] == "need alpha"
    assert updated["messages"][0]["content"].startswith("preface")
    assert updated["messages"][2]["content"].startswith("later")
    assert any(e.get("context_name") == "alpha" for e in events)
    assert not any("should-not" in t for t in expanded)
    assert not any("beta" in t for t in expanded)


@pytest.mark.asyncio
async def test_control_plane_rewrite_first_user_not_index_zero() -> None:
    body = {
        "system": "system text",
        "messages": [
            {"role": "assistant", "content": "preface should not be scanned"},
            {"role": "user", "content": "first user"},
            {"role": "user", "content": "later user"},
        ],
    }
    seen: List[str] = []

    async def rewrite_value(value, *, cc_version, manifest, available_context):
        if isinstance(value, str):
            seen.append(value)
            return value + "-rewritten", [{"id": "auto-memory"}], []
        if isinstance(value, dict) and value.get("role") == "user":
            seen.append(f"role:{value.get('role')}:{value.get('content')}")
            updated = dict(value)
            updated["content"] = str(value.get("content")) + "-rewritten"
            return updated, [], [{"id": "patch"}]
        seen.append(repr(value)[:40])
        return value, [], []

    with patch.object(
        cp, "_resolve_claude_prompt_patch_manifest_path", return_value=None
    ), patch.object(
        cp, "_load_claude_prompt_patch_manifest", return_value={}
    ), patch.object(
        cp, "_build_aawm_context_for_anthropic_request", return_value={}
    ), patch.object(
        cp, "_rewrite_claude_control_plane_in_value", side_effect=rewrite_value
    ), patch.object(
        cp,
        "_compact_claude_tool_advertisements_in_request_body",
        side_effect=lambda b, cc_version: (b, []),
    ), patch.object(
        cp,
        "_add_claude_system_prompt_override_logging_metadata",
        side_effect=lambda b, e: b,
    ), patch.object(
        cp, "_add_claude_prompt_patch_logging_metadata", side_effect=lambda b, e: b
    ):
        (
            updated,
            overrides,
            patches,
        ) = await cp.apply_claude_control_plane_rewrites_to_anthropic_request_body(
            body, {"cc_version": "1.0.0"}
        )

    assert updated["system"] == "system text-rewritten"
    assert updated["messages"][0]["content"] == "preface should not be scanned"
    assert updated["messages"][1]["content"] == "first user-rewritten"
    assert updated["messages"][2]["content"] == "later user"
    assert any(s.startswith("role:user:first user") for s in seen)
    assert not any("later user" in s for s in seen)
    assert not any("preface" in s for s in seen)
    assert overrides or patches


def test_stopword_filter_is_case_insensitive_for_acronyms() -> None:
    refs = cp._extract_aawm_dispatch_context_references("SQL json Api `KEEPME`")
    names = [name for name, _kind in refs]
    assert names == ["KEEPME"]


@pytest.mark.asyncio
async def test_request_wide_dispatch_lookup_budget_spans_text_blocks() -> None:
    """RR-053 #1: total fan-out is capped across all trusted text blocks, not per node."""
    # Two trusted surfaces with many distinct refs each; per-node max would
    # allow 24+24, but request-wide budget must stop earlier.
    block_a_names = [f"ALPHA{i}" for i in range(30)]
    block_b_names = [f"BETA{i}" for i in range(30)]
    system_text = "SubagentStart " + " ".join(f"`{n}`" for n in block_a_names)
    user_text = " ".join(f"`{n}`" for n in block_b_names)
    body = {
        "system": system_text,
        "messages": [{"role": "user", "content": user_text}],
        "litellm_metadata": {
            "claude_persisted_output_hooks": ["SubagentStart"],
        },
    }

    resolved: list[str] = []

    async def fake_resolve(
        name, available_context, placeholder_type="dispatch_backtick"
    ):
        resolved.append(name)
        return f"CONTENT:{name}", {
            "proc": "tristore_search_exact",
            "status": "resolved",
            "context_name": name,
            "placeholder_type": placeholder_type,
        }

    with patch.object(
        cp, "_resolve_aawm_context_reference", side_effect=fake_resolve
    ), patch.object(
        cp, "_expand_aawm_context_markers_in_text", side_effect=lambda t, c: (t, [])
    ), patch.object(
        cp, "_expand_aawm_dynamic_directives_in_text", side_effect=lambda t, c: (t, [])
    ), patch.object(
        cp,
        "_build_aawm_context_for_anthropic_request",
        return_value={"tenant": "t", "agent": "a"},
    ), patch.object(
        cp, "_add_aawm_dynamic_injection_logging_metadata", side_effect=lambda b, e: b
    ):
        # Shrink request budget for a tight assertion while keeping per-node high.
        with patch.object(
            cp, "_AAWM_DISPATCH_CONTEXT_REFERENCE_REQUEST_MAX", 10
        ), patch.object(cp, "_AAWM_DISPATCH_CONTEXT_REFERENCE_MAX", 24):
            (
                _updated,
                events,
            ) = await cp.expand_aawm_dynamic_directives_in_anthropic_request_body(body)

    assert len(resolved) == 10
    assert len(events) == 10
    # Budget consumed by the first trusted surface first (system before messages).
    assert all(name.startswith("ALPHA") for name in resolved)
    assert not any(name.startswith("BETA") for name in resolved)


@pytest.mark.asyncio
async def test_request_wide_budget_dedupes_names_across_blocks() -> None:
    """Duplicate identifiers across blocks should not double-spend budget."""
    body = {
        "system": "SubagentStart `SHARED` `ONLYA` `ONLYB`",
        "messages": [{"role": "user", "content": "`SHARED` `ONLYC` `ONLYD`"}],
        "litellm_metadata": {
            "claude_persisted_output_hooks": ["SubagentStart"],
        },
    }
    resolved: list[str] = []

    async def fake_resolve(
        name, available_context, placeholder_type="dispatch_backtick"
    ):
        resolved.append(name)
        return f"CONTENT:{name}", {
            "proc": "tristore_search_exact",
            "status": "resolved",
            "context_name": name,
            "placeholder_type": placeholder_type,
        }

    with patch.object(
        cp, "_resolve_aawm_context_reference", side_effect=fake_resolve
    ), patch.object(
        cp, "_expand_aawm_context_markers_in_text", side_effect=lambda t, c: (t, [])
    ), patch.object(
        cp, "_expand_aawm_dynamic_directives_in_text", side_effect=lambda t, c: (t, [])
    ), patch.object(
        cp, "_build_aawm_context_for_anthropic_request", return_value={"tenant": "t"}
    ), patch.object(
        cp, "_add_aawm_dynamic_injection_logging_metadata", side_effect=lambda b, e: b
    ):
        with patch.object(cp, "_AAWM_DISPATCH_CONTEXT_REFERENCE_REQUEST_MAX", 10):
            await cp.expand_aawm_dynamic_directives_in_anthropic_request_body(body)

    assert resolved.count("SHARED") == 1
    assert set(resolved) == {"SHARED", "ONLYA", "ONLYB", "ONLYC", "ONLYD"}


def test_proxy_shutdown_event_closes_dynamic_injection_pool_source() -> None:
    """RR-053 #2: shutdown path must reference the control-plane close helper."""
    from pathlib import Path

    source = Path("litellm/proxy/proxy_server.py").read_text(encoding="utf-8")
    assert "close_aawm_dynamic_injection_pool" in source
    assert "aawm_claude_control_plane" in source
    # Ensure it is invoked inside proxy_shutdown_event, not only imported elsewhere.
    shutdown_idx = source.find("async def proxy_shutdown_event")
    assert shutdown_idx != -1
    next_def = source.find("\nasync def ", shutdown_idx + 1)
    shutdown_body = source[shutdown_idx : next_def if next_def != -1 else None]
    assert "close_aawm_dynamic_injection_pool" in shutdown_body
