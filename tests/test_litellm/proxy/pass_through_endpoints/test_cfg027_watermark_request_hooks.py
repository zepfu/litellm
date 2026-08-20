"""CFG-027: request-path watermark intake/egress hooks.

HTTP-hook surface only. Detector/audit internals belong to CFG-026.
Direct ``/grok`` is out of scope (openai_passthrough only).

Request-path helpers are imported by planned names so this suite stays red
until CFG-027 installs them. Do not implement production watermark HTTP
hooks in this file.
"""

from __future__ import annotations

import ast
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException


ZWSP = "\u200b"
TOOL_ARG_SENTINEL = "CFG027_TOOL_ARG_BYTE_LOCK"
ENCRYPTED_SENTINEL = "CFG027_ENCRYPTED_REASONING_LOCK"
SCHEMA_SENTINEL = "CFG027_SCHEMA_DESCRIPTION_LOCK"

REPO_ROOT = Path(__file__).resolve().parents[4]
HANDLER_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/aawm_adapter_runtime/openai_passthrough_handler.py"
)
PASS_THROUGH_REQUEST_PATH = (
    REPO_ROOT / "litellm/proxy/pass_through_endpoints/pass_through_endpoints.py"
)
CODEX_DISPATCH_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/aawm_adapter_runtime/codex_dispatch.py"
)
CODEX_CANDIDATE_CALLS_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/aawm_adapter_runtime/codex_candidate_calls.py"
)
ANTHROPIC_ADAPTER_CALLS_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/aawm_adapter_runtime/anthropic_adapter_calls.py"
)
GROK_ROUTE_MODULE_PATH = (
    REPO_ROOT / "litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py"
)
SESSION_HISTORY_RECORD_PATH = (
    REPO_ROOT / "litellm/integrations/aawm_session_history/record.py"
)

INTAKE_MARKERS = (
    "apply_request_watermark_intake",
    "apply_watermark_policy",
    "watermark_input_audit",
)
EGRESS_MARKERS = (
    "apply_request_watermark_egress",
    "apply_watermark_policy",
    "watermark_input_audit",
)
COMPLETION_ADAPTER_SEND_SITES = (
    (
        ANTHROPIC_ADAPTER_CALLS_PATH,
        "_perform_anthropic_completion_adapter_messages_call",
        "litellm.acompletion",
    ),
    (
        ANTHROPIC_ADAPTER_CALLS_PATH,
        "_perform_normalized_anthropic_completion_adapter_stream",
        "litellm.acompletion",
    ),
    (
        CODEX_CANDIDATE_CALLS_PATH,
        "_perform_codex_cohere_chat_completions_adapter_call",
        "litellm.acompletion",
    ),
    (
        CODEX_CANDIDATE_CALLS_PATH,
        "_perform_codex_kimi_chat_completions_adapter_call",
        "litellm.acompletion",
    ),
    (
        CODEX_CANDIDATE_CALLS_PATH,
        "_perform_codex_alibaba_token_plan_adapter_call",
        "litellm.acompletion",
    ),
    (
        CODEX_CANDIDATE_CALLS_PATH,
        "_perform_opencode_zen_completion_call",
        "litellm.acompletion",
    ),
    (
        CODEX_CANDIDATE_CALLS_PATH,
        "_perform_codex_auto_agent_openrouter_completion_request",
        "litellm.acompletion",
    ),
)


def _load_text_watermark_config(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.config import (
        load_text_watermark_config,
    )

    return load_text_watermark_config(*args, **kwargs)


def _apply_request_watermark_intake(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.policy import (
        apply_request_watermark_intake,
    )

    return apply_request_watermark_intake(*args, **kwargs)


def _apply_request_watermark_egress(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.policy import (
        apply_request_watermark_egress,
    )

    return apply_request_watermark_egress(*args, **kwargs)


def _build_watermark_policy_failure_session_history_record(
    *args: Any, **kwargs: Any
) -> Any:
    from litellm.integrations.aawm_session_history.record import (
        _build_watermark_policy_failure_session_history_record,
    )

    return _build_watermark_policy_failure_session_history_record(*args, **kwargs)


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def _optional_field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} not found in {path}")


def _first_marker_index(source: str, markers: tuple[str, ...]) -> int:
    indexes = [source.find(marker) for marker in markers if marker in source]
    assert indexes, f"expected one of {markers} in source"
    return min(indexes)


def _detect_config() -> Any:
    return _load_text_watermark_config(
        {
            "mode": "detect",
            "unicode": {
                "enabled": True,
                "policy": "conservative",
                "normalize_spaces": True,
                "nfkc": False,
            },
            "removal": {
                "enabled": False,
                "stream_policy": "audit_only",
                "on_unremovable": "allow",
            },
            "statistical_detectors": [],
        }
    )


def _sanitize_config() -> Any:
    return _load_text_watermark_config(
        {
            "mode": "sanitize",
            "unicode": {
                "enabled": True,
                "policy": "conservative",
                "normalize_spaces": True,
                "nfkc": False,
            },
            "removal": {
                "enabled": True,
                "stream_policy": "audit_only",
                "on_unremovable": "allow",
            },
            "statistical_detectors": [],
        }
    )


def _enforce_config() -> Any:
    return _load_text_watermark_config(
        {
            "mode": "enforce",
            "unicode": {
                "enabled": True,
                "policy": "conservative",
                "normalize_spaces": True,
                "nfkc": False,
            },
            "removal": {
                "enabled": True,
                "stream_policy": "buffer_response",
                "on_unremovable": "block",
            },
            "statistical_detectors": [],
        }
    )


def _off_config() -> Any:
    return _load_text_watermark_config(None)


def _responses_request_body(*, carrier: bool = False) -> dict[str, Any]:
    visible = f"user{ZWSP}question" if carrier else "Please inspect the pantry layout."
    instructions = (
        f"Follow the user.{ZWSP}" if carrier else "Follow the user."
    )
    return {
        "model": "gpt-5.4",
        "instructions": instructions,
        "input": [
            {
                "type": "message",
                "role": "user",
                "id": "msg_visible",
                "content": [
                    {
                        "type": "input_text",
                        "text": visible,
                    }
                ],
            },
            {
                "type": "function_call",
                "id": "fc_keep_id",
                "call_id": "call_keep_id",
                "name": "bash",
                "arguments": json.dumps(
                    {"cmd": "pwd", "note": TOOL_ARG_SENTINEL},
                    separators=(",", ":"),
                ),
            },
            {
                "type": "reasoning",
                "id": "rs_keep_id",
                "encrypted_content": ENCRYPTED_SENTINEL,
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": SCHEMA_SENTINEL,
                        }
                    },
                },
            }
        ],
    }


def _audit_from_result(result: Any, metadata: Mapping[str, Any] | None = None) -> Any:
    if metadata is not None and metadata.get("watermark_input_audit") is not None:
        return metadata["watermark_input_audit"]
    if result is None:
        return None
    for name in ("watermark_input_audit", "audit"):
        value = _optional_field(result, name)
        if value is not None:
            return value
    nested = _optional_field(result, "metadata")
    if isinstance(nested, Mapping):
        return nested.get("watermark_input_audit")
    return None


def _stage(audit: Any, name: str) -> Any:
    if audit is None:
        return None
    direct = _optional_field(audit, name)
    if direct is not None:
        return direct
    stages = _optional_field(audit, "stages")
    if isinstance(stages, Mapping):
        return stages.get(name)
    return None


def _signal_detected(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, Mapping) or hasattr(value, "signal_detected"):
        return bool(_field(value, "signal_detected"))
    return False


# ---------------------------------------------------------------------------
# Source inspection: intake after JSON parse
# ---------------------------------------------------------------------------


def test_intake_runs_after_json_parse_in_base_openai_pass_through_handler() -> None:
    source = _function_source(HANDLER_PATH, "_base_openai_pass_through_handler")
    parse_idx = source.find("get_request_body_fn")
    assert parse_idx != -1
    intake_idx = _first_marker_index(source, INTAKE_MARKERS)
    assert "watermark_input_audit" in source or "apply_request_watermark_intake" in source
    assert "apply_watermark_policy" in source or "apply_request_watermark_intake" in source
    assert intake_idx > parse_idx
    dispatch_idx = source.find("try_dispatch_codex_request_fn")
    assert dispatch_idx != -1
    assert intake_idx < dispatch_idx


# ---------------------------------------------------------------------------
# Source inspection: egress immediately before every outbound send
# ---------------------------------------------------------------------------


def test_pass_through_request_runs_egress_immediately_before_outbound_send() -> None:
    source = _function_source(PASS_THROUGH_REQUEST_PATH, "pass_through_request")
    egress_idx = _first_marker_index(source, EGRESS_MARKERS)
    assert "apply_request_watermark_egress" in source
    assert "watermark_input_audit" in source
    for send_name in (
        "_send_stream_pre_first_byte",
        "_send_non_stream_pre_first_byte",
        "async_client.send",
        "non_streaming_http_request_handler",
    ):
        send_idx = source.find(send_name)
        assert send_idx != -1, send_name
        assert egress_idx < send_idx, send_name


def test_try_dispatch_codex_request_runs_egress_on_early_return_path() -> None:
    source = _function_source(CODEX_DISPATCH_PATH, "try_dispatch_codex_request")
    egress_idx = _first_marker_index(source, EGRESS_MARKERS)
    assert "apply_request_watermark_egress" in source
    first_return_handler = min(
        idx
        for name in (
            "_handle_codex_auto_agent_alias_route",
            "_handle_codex_opencode_zen_adapter_route",
            "_handle_codex_kimi_chat_completions_adapter_route",
            "_handle_codex_alibaba_token_plan_adapter_route",
        )
        if (idx := source.find(name)) != -1
    )
    assert egress_idx < first_return_handler


@pytest.mark.parametrize(
    ("path", "function_name", "send_token"),
    COMPLETION_ADAPTER_SEND_SITES,
)
def test_completion_adapter_acompletion_owners_run_egress_before_send(
    path: Path,
    function_name: str,
    send_token: str,
) -> None:
    source = _function_source(path, function_name)
    egress_idx = _first_marker_index(source, EGRESS_MARKERS)
    send_idx = source.find(send_token)
    assert send_idx != -1, send_token
    assert "apply_request_watermark_egress" in source
    assert egress_idx < send_idx


def test_no_known_outbound_send_site_skips_watermark_egress() -> None:
    missing: list[str] = []
    for path, function_name, send_token in (
        (PASS_THROUGH_REQUEST_PATH, "pass_through_request", "async_client.send"),
        (CODEX_DISPATCH_PATH, "try_dispatch_codex_request", "_handle_codex_"),
        *COMPLETION_ADAPTER_SEND_SITES,
    ):
        source = _function_source(path, function_name)
        if "apply_request_watermark_egress" not in source:
            missing.append(f"{path.name}:{function_name}")
            continue
        if send_token not in source:
            missing.append(f"{path.name}:{function_name}:missing:{send_token}")
    assert missing == []


# ---------------------------------------------------------------------------
# detect mode
# ---------------------------------------------------------------------------


def test_detect_mode_scans_harness_original_and_upstream_sent_without_mutation() -> None:
    original = _responses_request_body(carrier=True)
    harness = copy.deepcopy(original)
    provider_bound = copy.deepcopy(original)
    metadata: dict[str, Any] = {}
    litellm_metadata: dict[str, Any] = {}
    config = _detect_config()

    intake = _apply_request_watermark_intake(
        body=harness,
        config=config,
        endpoint="responses",
        direction="request",
    )
    egress = _apply_request_watermark_egress(
        body=provider_bound,
        intake=intake,
        config=config,
        endpoint="responses",
        direction="request",
        metadata=metadata,
        litellm_metadata=litellm_metadata,
    )

    assert harness == original
    assert provider_bound == original
    assert ZWSP in harness["instructions"]
    assert ZWSP in harness["input"][0]["content"][0]["text"]
    assert ZWSP in provider_bound["instructions"]
    assert ZWSP in provider_bound["input"][0]["content"][0]["text"]
    assert provider_bound["input"][1]["arguments"] == original["input"][1]["arguments"]

    audit = _audit_from_result(egress, metadata)
    assert audit is not None
    harness_stage = _stage(audit, "harness_original")
    sent_stage = _stage(audit, "upstream_sent")
    assert harness_stage is not None
    assert sent_stage is not None
    assert _signal_detected(harness_stage) is True
    assert _signal_detected(sent_stage) is True
    assert _signal_detected(audit) is True
    assert metadata["watermark_input_audit"] is audit
    assert litellm_metadata["watermark_input_audit"] is audit

    from litellm.integrations.aawm_session_history.record import (
        _build_session_history_metadata,
    )

    copied = _build_session_history_metadata(
        metadata=metadata,
        request_tags=[],
        tenant_id=None,
    )
    assert copied["watermark_input_audit"]["signal_detected"] is True
    assert "watermark_output_audit" not in copied


# ---------------------------------------------------------------------------
# sanitize + removal.enabled
# ---------------------------------------------------------------------------


def test_sanitize_mutates_only_visible_nodes_on_provider_bound_body() -> None:
    original = _responses_request_body(carrier=True)
    harness = copy.deepcopy(original)
    provider_bound = copy.deepcopy(original)
    metadata: dict[str, Any] = {}
    config = _sanitize_config()

    intake = _apply_request_watermark_intake(
        body=harness,
        config=config,
        endpoint="responses",
        direction="request",
    )
    egress = _apply_request_watermark_egress(
        body=provider_bound,
        intake=intake,
        config=config,
        endpoint="responses",
        direction="request",
        metadata=metadata,
        litellm_metadata={},
    )
    out_body = _optional_field(egress, "body", provider_bound)
    assert isinstance(out_body, dict)

    assert ZWSP in harness["instructions"]
    assert ZWSP in harness["input"][0]["content"][0]["text"]
    assert ZWSP not in out_body["instructions"]
    assert ZWSP not in out_body["input"][0]["content"][0]["text"]
    assert out_body["input"][1]["arguments"] == original["input"][1]["arguments"]
    assert out_body["input"][2]["encrypted_content"] == ENCRYPTED_SENTINEL
    assert out_body["tools"] == original["tools"]
    assert (
        out_body["input"][1]["id"],
        out_body["input"][1]["call_id"],
        out_body["input"][2]["id"],
    ) == (
        original["input"][1]["id"],
        original["input"][1]["call_id"],
        original["input"][2]["id"],
    )

    audit = _audit_from_result(egress, metadata)
    assert audit is not None
    harness_stage = _stage(audit, "harness_original")
    sent_stage = _stage(audit, "upstream_sent")
    assert _signal_detected(harness_stage) is True
    assert _signal_detected(sent_stage) is False


# ---------------------------------------------------------------------------
# enforce blocks before egress + primary session_history failure row
# ---------------------------------------------------------------------------


def test_enforce_blocks_before_egress_with_blocked_input_audit() -> None:
    original = _responses_request_body(carrier=True)
    provider_bound = copy.deepcopy(original)
    metadata: dict[str, Any] = {"session_id": "sess-cfg027-enforce"}
    litellm_metadata: dict[str, Any] = {}
    config = _enforce_config()
    intake = _apply_request_watermark_intake(
        body=copy.deepcopy(original),
        config=config,
        endpoint="responses",
        direction="request",
    )

    with pytest.raises(HTTPException) as exc_info:
        _apply_request_watermark_egress(
            body=provider_bound,
            intake=intake,
            config=config,
            endpoint="responses",
            direction="request",
            metadata=metadata,
            litellm_metadata=litellm_metadata,
        )

    assert exc_info.value.status_code in {400, 403, 409, 422}
    audit = metadata.get("watermark_input_audit")
    if audit is None:
        detail = exc_info.value.detail
        if isinstance(detail, Mapping):
            audit = detail.get("watermark_input_audit")
    assert audit is not None
    assert _field(audit, "status") == "blocked"
    assert ZWSP in provider_bound["instructions"]


def test_enforce_block_has_primary_session_history_failure_row_path() -> None:
    record_source = SESSION_HISTORY_RECORD_PATH.read_text(encoding="utf-8")
    assert "_build_watermark_policy_failure_session_history_record" in record_source
    failure_event_source = _function_source(
        SESSION_HISTORY_RECORD_PATH,
        "_handle_session_history_failure_event",
    )
    observation_source = _function_source(
        SESSION_HISTORY_RECORD_PATH,
        "_build_failure_observation_only_record",
    )
    assert (
        "_build_watermark_policy_failure_session_history_record" in failure_event_source
        or "_build_watermark_policy_failure_session_history_record"
        in observation_source
    )
    assert "watermark_input_audit" in record_source

    audit = {
        "schema_version": 1,
        "direction": "request",
        "mode": "enforce",
        "status": "blocked",
        "signal_detected": True,
        "confirmed_watermark_detected": False,
        "vendor_attribution": "unknown",
    }
    kwargs = {
        "litellm_params": {
            "metadata": {
                "session_id": "sess-cfg027-enforce",
                "watermark_input_audit": audit,
                "source_status": "failure",
            }
        }
    }
    record = _build_watermark_policy_failure_session_history_record(
        kwargs,
        HTTPException(status_code=403, detail={"watermark_input_audit": audit}),
        start_time=None,
        end_time=None,
    )
    assert record is not None
    assert record.get("_skip_session_history") is not True
    history_metadata = record.get("metadata") or {}
    assert history_metadata.get("source_status") == "failure"
    assert _field(history_metadata.get("watermark_input_audit") or audit, "status") == (
        "blocked"
    )


# ---------------------------------------------------------------------------
# mode off
# ---------------------------------------------------------------------------


def test_mode_off_adds_neither_audit_key() -> None:
    original = _responses_request_body(carrier=True)
    harness = copy.deepcopy(original)
    provider_bound = copy.deepcopy(original)
    metadata: dict[str, Any] = {}
    litellm_metadata: dict[str, Any] = {}
    config = _off_config()

    intake = _apply_request_watermark_intake(
        body=harness,
        config=config,
        endpoint="responses",
        direction="request",
    )
    egress = _apply_request_watermark_egress(
        body=provider_bound,
        intake=intake,
        config=config,
        endpoint="responses",
        direction="request",
        metadata=metadata,
        litellm_metadata=litellm_metadata,
    )

    assert harness == original
    assert provider_bound == original
    assert _audit_from_result(egress, metadata) is None
    assert "watermark_input_audit" not in metadata
    assert "watermark_output_audit" not in metadata
    assert "watermark_input_audit" not in litellm_metadata
    assert "watermark_output_audit" not in litellm_metadata
    if isinstance(egress, Mapping):
        assert "watermark_input_audit" not in egress
        assert "watermark_output_audit" not in egress


# ---------------------------------------------------------------------------
# grok_proxy_route is out of scope
# ---------------------------------------------------------------------------


def test_grok_proxy_route_does_not_require_watermark_request_hooks() -> None:
    source = GROK_ROUTE_MODULE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    grok_fn = next(
        node
        for node in module.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "grok_proxy_route"
    )
    grok_source = ast.get_source_segment(source, grok_fn) or ""
    assert grok_source
    for marker in (
        "apply_request_watermark_intake",
        "apply_request_watermark_egress",
        "watermark_input_audit",
        "aawm_text_watermark",
    ):
        assert marker not in grok_source
    assert "watermark" not in grok_source.lower()
