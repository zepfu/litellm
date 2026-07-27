"""Wave 6D module-local tests for persisted-output request policy."""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from litellm.proxy.pass_through_endpoints.aawm_request_policy import (
    persisted_output as po,
)

TARGET_PATH = Path(po.__file__).resolve()
EXPECTED_SYMBOLS = (
    "_is_claude_persisted_output_expansion_enabled",
    "_get_claude_persisted_output_root",
    "_resolve_claude_persisted_output_path",
    "_build_claude_persisted_output_source_metadata",
    "_compact_google_adapter_persisted_output_preview_and_expanded_text",
    "_compact_expanded_claude_persisted_output_text_for_google_adapter",
    "_compact_google_adapter_text_part_sequence",
    "_compact_google_adapter_followup_request_contents",
    "_compact_google_adapter_persisted_output_value",
    "_compact_google_adapter_persisted_output_in_anthropic_request_body",
    "_expand_claude_persisted_output_text",
    "_expand_claude_persisted_output_value",
    "_expand_claude_persisted_output_in_anthropic_request_body",
    "_estimate_google_content_text_chars",
)


def _load_direct_owner_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_wave6d_direct_owner_persisted_output",
        TARGET_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load direct-owner module from {TARGET_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", autouse=True)
def _isolate_direct_owner_module() -> Iterator[None]:
    global po
    canonical_module = po
    po = _load_direct_owner_module()
    try:
        yield
    finally:
        po = canonical_module


def _persisted_output_marker(path: Path, hook: str = "SubagentStart") -> str:
    return (
        "<system-reminder>\n"
        f"{hook} hook additional context: <persisted-output>\n"
        f"Output too large (24.4KB). Full output saved to: {path}\n\n"
        "Preview (first 2KB):\n"
        "truncated preview\n"
        "</persisted-output>\n"
        "</system-reminder>\n"
    )


def _create_persisted_file(
    tmp_path: Path,
    *,
    name: str = "hook-1-additionalContext.txt",
    text: str = "persisted full output",
) -> tuple[Path, Path]:
    root = tmp_path / ".claude" / "projects"
    output_path = root / "project" / "session" / "tool-results" / name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return root, output_path


def test_module_owns_exact_persisted_output_symbol_inventory() -> None:
    tree = ast.parse(TARGET_PATH.read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert tuple(po._HOST_FUNCTION_NAMES) == EXPECTED_SYMBOLS
    assert set(EXPECTED_SYMBOLS) <= defined
    assert "_compact_openai_adapter_claude_context_text" not in defined
    assert "_add_claude_persisted_output_logging_metadata" not in defined


def test_module_does_not_import_god_module() -> None:
    tree = ast.parse(TARGET_PATH.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    assert not [
        module
        for module in imported_modules
        if "llm_passthrough_endpoints" in module
    ]


def test_install_preserves_same_object_facades_and_host_monkeypatches(
) -> None:
    originals = {name: getattr(po, name) for name in EXPECTED_SYMBOLS}
    host_globals = dict(vars(po))
    try:
        po.install(host_globals)
        for name in EXPECTED_SYMBOLS:
            assert host_globals[name] is getattr(po, name)

        host_globals["_is_claude_persisted_output_expansion_enabled"] = (
            lambda: False
        )
        original = "not a persisted output marker"
        assert po._expand_claude_persisted_output_text(original) == (
            original,
            False,
            None,
            None,
        )
    finally:
        for name, original_function in originals.items():
            setattr(po, name, original_function)


@pytest.mark.parametrize("enabled_value", ["1", "true", "yes", "on", "TRUE"])
def test_expands_direct_persisted_output_and_builds_source_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled_value: str,
) -> None:
    root, output_path = _create_persisted_file(
        tmp_path,
        text="line one\nline two\n",
    )
    monkeypatch.setenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", enabled_value)
    monkeypatch.setenv("LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", str(root))

    expanded, changed, hook, metadata = (
        po._expand_claude_persisted_output_text(
            _persisted_output_marker(output_path)
        )
    )

    assert changed is True
    assert hook == "subagentstart"
    assert expanded == (
        "<system-reminder>\n"
        "SubagentStart hook additional context: <persisted-output>\n"
        "line one\nline two\n"
        "</persisted-output>\n"
        "</system-reminder>\n"
    )
    assert metadata is not None
    assert metadata["path"] == str(output_path)
    assert metadata["basename"] == output_path.name
    assert metadata["bytes"] == len("line one\nline two".encode())
    assert len(metadata["content_hash"]) == 64


def test_expands_nested_continuation_output_items_without_mutating_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, first_path = _create_persisted_file(
        tmp_path,
        name="first-additionalContext.txt",
        text="first full output",
    )
    _, second_path = _create_persisted_file(
        tmp_path,
        name="second-additionalContext.txt",
        text="second full output",
    )
    monkeypatch.setenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "1")
    monkeypatch.setenv("LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", str(root))
    unchanged_item = {"type": "input_text", "text": "continuation state"}
    value: dict[str, Any] = {
        "previous_response_id": "resp_previous",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "text",
                        "text": _persisted_output_marker(first_path),
                    },
                    unchanged_item,
                ],
            },
            {
                "type": "function_call_output",
                "output": {
                    "type": "text",
                    "text": _persisted_output_marker(
                        second_path,
                        hook="SessionStart",
                    ),
                },
            },
        ],
    }

    updated, count, hooks, sources = po._expand_claude_persisted_output_value(
        value
    )

    assert updated is not value
    assert value["output"][0]["content"][0]["text"].endswith(
        "truncated preview\n</persisted-output>\n</system-reminder>\n"
    )
    assert updated["previous_response_id"] == "resp_previous"
    assert updated["output"][0]["content"][1] is unchanged_item
    assert "first full output" in updated["output"][0]["content"][0]["text"]
    assert "second full output" in updated["output"][1]["output"]["text"]
    assert count == 2
    assert hooks == {"subagentstart", "sessionstart"}
    assert [source["basename"] for source in sources] == [
        first_path.name,
        second_path.name,
    ]


@pytest.mark.parametrize(
    "enabled,path_kind",
    [
        (False, "valid"),
        (True, "outside"),
        (True, "wrong-directory"),
        (True, "wrong-name"),
        (True, "missing"),
    ],
)
def test_expansion_edge_cases_are_noops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    path_kind: str,
) -> None:
    root, valid_path = _create_persisted_file(tmp_path)
    candidate = valid_path
    if path_kind == "outside":
        candidate = tmp_path / "outside-additionalContext.txt"
        candidate.write_text("outside", encoding="utf-8")
    elif path_kind == "wrong-directory":
        candidate = root / "project" / "session" / "wrong" / valid_path.name
        candidate.parent.mkdir(parents=True)
        candidate.write_text("wrong directory", encoding="utf-8")
    elif path_kind == "wrong-name":
        candidate = valid_path.with_name("wrong.txt")
        candidate.write_text("wrong name", encoding="utf-8")
    elif path_kind == "missing":
        candidate = valid_path.with_name("missing-additionalContext.txt")

    monkeypatch.setenv(
        "LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT",
        "1" if enabled else "0",
    )
    monkeypatch.setenv("LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", str(root))
    marker = _persisted_output_marker(candidate)

    assert po._expand_claude_persisted_output_text(marker) == (
        marker,
        False,
        None,
        None,
    )


def test_file_read_exceptions_are_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker_path = Path(
        "/allowed/tool-results/hook-additionalContext.txt"
    )
    marker = _persisted_output_marker(marker_path)
    failing_path = SimpleNamespace(
        read_text=lambda **_kwargs: (_ for _ in ()).throw(OSError("read failed"))
    )
    monkeypatch.setattr(
        po,
        "_is_claude_persisted_output_expansion_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        po,
        "_resolve_claude_persisted_output_path",
        lambda _path: failing_path,
    )

    assert po._expand_claude_persisted_output_text(marker) == (
        marker,
        False,
        None,
        None,
    )


def test_google_delegate_compacts_split_output_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        po,
        "_get_google_adapter_persisted_output_char_cap",
        lambda: 40,
    )
    monkeypatch.setattr(
        po,
        "_get_google_adapter_auxiliary_context_char_cap",
        lambda: 400,
    )
    parts: list[Any] = [
        {"type": "text", "text": "<system-reminder>\n"},
        {
            "type": "text",
            "text": (
                "SubagentStart hook additional context: "
                "<persisted-output>\n" + ("A" * 300)
            ),
        },
        {
            "type": "text",
            "text": (
                "\n</persisted-output>\n"
                "</system-reminder>\n"
            ),
        },
        {"type": "image", "source": "unchanged"},
    ]

    updated, count, hooks, metadata, changed = (
        po._compact_google_adapter_text_part_sequence(parts)
    )

    assert changed is True
    assert count == 1
    assert hooks == {"subagentstart"}
    assert len(updated) == 2
    assert "Gemini adapter compacted persisted-output" in updated[0]["text"]
    assert updated[1] is parts[3]
    assert metadata[0]["original_chars"] == 300
    assert metadata[0]["kept_chars"] <= 40


def test_google_followup_continuation_compacts_in_place_and_reports_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        po,
        "_get_google_adapter_followup_persisted_output_char_cap",
        lambda: 30,
    )
    monkeypatch.setattr(
        po,
        "_get_google_adapter_followup_auxiliary_context_char_cap",
        lambda: 400,
    )
    request_block: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "<system-reminder>\n"
                            "SessionStart hook additional context: "
                            "<persisted-output>\n"
                            + ("B" * 300)
                            + "\n</persisted-output>\n"
                            "</system-reminder>\n"
                        )
                    }
                ],
            },
            {
                "role": "model",
                "parts": [{"text": "prior continuation output"}],
            },
        ]
    }

    changes = po._compact_google_adapter_followup_request_contents(
        request_block
    )

    compacted_text = request_block["contents"][0]["parts"][0]["text"]
    assert "Gemini adapter compacted persisted-output" in compacted_text
    assert changes["followup_persisted_output_compacted_count"] == 1
    assert changes["followup_persisted_output_hooks"] == ["sessionstart"]
    assert changes["followup_persisted_output_char_cap"] == 30
    assert changes["followup_persisted_output_text_chars_after"] < (
        changes["followup_persisted_output_text_chars_before"]
    )
    assert request_block["contents"][1]["parts"][0]["text"] == (
        "prior continuation output"
    )


@pytest.mark.parametrize(
    "request_block",
    [{}, {"contents": []}, {"contents": [{"role": "user", "parts": []}]}],
)
def test_google_followup_noop_edges_return_empty_changes(
    request_block: dict[str, Any],
) -> None:
    original = dict(request_block)

    assert (
        po._compact_google_adapter_followup_request_contents(request_block)
        == {}
    )
    assert request_block == original


def test_google_delegate_exceptions_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExpectedError(RuntimeError):
        pass

    def raise_expected(
        _text: str,
        *,
        cap: int,
    ) -> tuple[str, int, set[str], list[dict[str, Any]]]:
        raise ExpectedError(str(cap))

    monkeypatch.setattr(
        po._anthropic_google_shaping,
        "_compact_google_adapter_persisted_output_preview_and_expanded_text",
        raise_expected,
    )

    with pytest.raises(ExpectedError, match="17"):
        po._compact_google_adapter_persisted_output_preview_and_expanded_text(
            "payload",
            cap=17,
        )


# ---------------------------------------------------------------------------
# Body-level expansion contract
# ---------------------------------------------------------------------------


def test_body_expansion_calls_logging_callback_with_correct_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, output_path = _create_persisted_file(tmp_path, text="body output")
    monkeypatch.setenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "1")
    monkeypatch.setenv("LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", str(root))

    call_log: list[Any] = []

    def fake_logging_callback(
        body: dict[str, Any],
        expanded_count: int,
        hooks: set[str],
        source_metadata_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        call_log.append(
            (body, expanded_count, hooks, source_metadata_items)
        )
        body["litellm_metadata"] = {"injected": True}
        return body

    monkeypatch.setattr(
        po, "_persisted_output_logging_callback", fake_logging_callback
    )

    request_body: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _persisted_output_marker(output_path),
                    }
                ],
            }
        ]
    }

    updated, count, hooks, sources = (
        po._expand_claude_persisted_output_in_anthropic_request_body(
            request_body
        )
    )

    assert count == 1
    assert hooks == {"subagentstart"}
    assert len(sources) == 1
    assert updated["litellm_metadata"] == {"injected": True}
    assert len(call_log) == 1
    _, cb_count, cb_hooks, cb_sources = call_log[0]
    assert cb_count == 1
    assert cb_hooks == {"subagentstart"}
    assert len(cb_sources) == 1


def test_body_expansion_noop_when_nothing_expanded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "0")
    call_log: list[Any] = []
    monkeypatch.setattr(
        po,
        "_persisted_output_logging_callback",
        lambda *a: call_log.append(a) or {},
    )

    body: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}]}
    updated, count, hooks, sources = (
        po._expand_claude_persisted_output_in_anthropic_request_body(body)
    )

    assert count == 0
    assert hooks == set()
    assert sources == []
    assert updated is body
    assert call_log == []


def test_body_expansion_skips_callback_when_not_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, output_path = _create_persisted_file(tmp_path, text="no cb")
    monkeypatch.setenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "1")
    monkeypatch.setenv("LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", str(root))
    monkeypatch.setattr(po, "_persisted_output_logging_callback", None)

    body: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _persisted_output_marker(output_path)}
                ],
            }
        ]
    }

    updated, count, hooks, sources = (
        po._expand_claude_persisted_output_in_anthropic_request_body(body)
    )

    assert count == 1
    assert "litellm_metadata" not in updated


def test_body_expansion_callback_four_arg_signature_and_timing_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: preserve the callback signature and span timing sequence.

    The configured observability callback
    (_add_claude_persisted_output_logging_metadata) accepts four positional
    arguments. Timing is attached to its span descriptor after it returns.
    """
    root, output_path = _create_persisted_file(tmp_path, text="regression")
    monkeypatch.setenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "1")
    monkeypatch.setenv("LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", str(root))

    call_args: list[Any] = []
    events: list[str] = []
    span_started_at = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
    span_ended_at = datetime(2026, 7, 27, 12, 0, 1, tzinfo=timezone.utc)
    timestamps = iter((span_started_at, span_ended_at))

    class SequencedDateTime:
        @classmethod
        def now(cls, tz: Any) -> datetime:
            value = next(timestamps)
            assert tz is timezone.utc
            events.append(
                "now:start" if value is span_started_at else "now:end"
            )
            return value

    def format_timestamp(value: datetime) -> str:
        events.append(
            "format:start" if value is span_started_at else "format:end"
        )
        return value.isoformat().replace("+00:00", "Z")

    def strict_four_arg_callback(
        body: dict[str, Any],
        expanded_count: int,
        hooks: set[str],
        source_metadata_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        events.append("callback")
        call_args.append((body, expanded_count, hooks, source_metadata_items))
        body["litellm_metadata"] = {
            "langfuse_spans": [
                {"name": "other"},
                {"name": "claude.persisted_output_expand"},
            ]
        }
        return body

    monkeypatch.setattr(po, "datetime", SequencedDateTime)
    monkeypatch.setattr(
        po,
        "_format_langfuse_span_timestamp",
        format_timestamp,
        raising=False,
    )
    monkeypatch.setattr(
        po, "_persisted_output_logging_callback", strict_four_arg_callback
    )

    body: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _persisted_output_marker(output_path)}
                ],
            }
        ]
    }
    updated, count, hooks, sources = (
        po._expand_claude_persisted_output_in_anthropic_request_body(body)
    )

    assert count == 1
    assert hooks == {"subagentstart"}
    assert len(sources) == 1
    assert len(call_args) == 1
    _, cb_count, cb_hooks, cb_sources = call_args[0]
    assert cb_count == 1
    assert cb_hooks == {"subagentstart"}
    assert len(cb_sources) == 1
    spans = updated["litellm_metadata"]["langfuse_spans"]
    assert spans[0] == {"name": "other"}
    assert spans[1] == {
        "name": "claude.persisted_output_expand",
        "start_time": "2026-07-27T12:00:00Z",
        "end_time": "2026-07-27T12:00:01Z",
    }
    assert events == [
        "now:start",
        "callback",
        "format:start",
        "now:end",
        "format:end",
    ]


# ---------------------------------------------------------------------------
# Installed/direct parity
# ---------------------------------------------------------------------------


def test_estimate_google_content_text_chars_direct_parity() -> None:
    """Direct module function matches the canonical algorithm."""
    block = {"parts": [{"text": "hello"}, {"text": "world"}, {"image": True}]}
    assert po._estimate_google_content_text_chars(block) == 10
    assert po._estimate_google_content_text_chars({}) == 0
    assert po._estimate_google_content_text_chars("not a dict") == 0
    assert po._estimate_google_content_text_chars({"parts": "bad"}) == 0
    assert po._estimate_google_content_text_chars({"parts": [None, 42]}) == 0


def test_estimate_google_content_text_chars_installed_parity(
) -> None:
    """Installed host facade produces same results as direct call."""
    originals = {name: getattr(po, name) for name in po._HOST_FUNCTION_NAMES}
    host_globals = dict(vars(po))
    try:
        po.install(host_globals)
        block = {"parts": [{"text": "abc"}, {"text": "defgh"}]}
        direct = po._estimate_google_content_text_chars(block)
        installed = host_globals["_estimate_google_content_text_chars"](block)
        assert direct == installed == 8
    finally:
        for name, original_function in originals.items():
            setattr(po, name, original_function)


# ---------------------------------------------------------------------------
# Callback ordering
# ---------------------------------------------------------------------------


def test_body_expansion_callback_order_expand_then_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expansion happens before the logging callback is invoked."""
    root, output_path = _create_persisted_file(tmp_path, text="order check")
    monkeypatch.setenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "1")
    monkeypatch.setenv("LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", str(root))

    order: list[str] = []
    original_expand_value = po._expand_claude_persisted_output_value

    def tracking_expand_value(value: Any) -> Any:
        order.append("expand")
        return original_expand_value(value)

    def tracking_callback(
        body: dict[str, Any],
        _c: int,
        _h: set[str],
        _s: list[dict[str, Any]],
    ) -> dict[str, Any]:
        order.append("log")
        return body

    monkeypatch.setattr(
        po, "_expand_claude_persisted_output_value", tracking_expand_value
    )
    monkeypatch.setattr(
        po, "_persisted_output_logging_callback", tracking_callback
    )

    body: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _persisted_output_marker(output_path)}
                ],
            }
        ]
    }
    po._expand_claude_persisted_output_in_anthropic_request_body(body)

    assert order[-1] == "log"
    assert all(entry == "expand" for entry in order[:-1])
    assert len(order) >= 2
