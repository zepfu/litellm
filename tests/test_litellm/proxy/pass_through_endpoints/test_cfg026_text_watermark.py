"""CFG-026: Unicode-carrier detector and bounded watermark audits.

Unit surface only. Request/response HTTP hooks belong to CFG-027/028.
Tests import ``aawm_text_watermark`` modules that do not exist yet (red phase).
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError


def _load_text_watermark_config(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.config import (
        load_text_watermark_config,
    )

    return load_text_watermark_config(*args, **kwargs)


def _apply_watermark_policy(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.policy import (
        apply_watermark_policy,
    )

    return apply_watermark_policy(*args, **kwargs)


def _evaluate_statistical_detectors(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.policy import (
        evaluate_statistical_detectors,
    )

    return evaluate_statistical_detectors(*args, **kwargs)


def _extract_visible_text_nodes(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.text_nodes import (
        extract_visible_text_nodes,
    )

    return extract_visible_text_nodes(*args, **kwargs)


def _detect_unicode_carriers(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.unicode_detector import (
        detect_unicode_carriers,
    )

    return detect_unicode_carriers(*args, **kwargs)


def _sanitize_unicode_carriers(*args: Any, **kwargs: Any) -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.unicode_detector import (
        sanitize_unicode_carriers,
    )

    return sanitize_unicode_carriers(*args, **kwargs)


ZWSP = "\u200b"
ZWJ = "\u200d"
VS16 = "\ufe0f"
NONCHARACTER = "\ufffe"
EXOTIC_EN_QUAD = "\u2000"
EXOTIC_IDEOGRAPHIC_SPACE = "\u3000"
# Context-valid emoji ZWJ sequence (man + ZWJ + woman + ZWJ + girl).
EMOJI_FAMILY = "👨‍👩‍👧"
# White flag + VS16 + ZWJ + rainbow; valid emoji glue, not a carrier.
EMOJI_RAINBOW_FLAG = "🏳️‍🌈"

PROMPT_SENTINEL = "CFG026_FULL_PROMPT_SENTINEL_DO_NOT_PERSIST"
TOOL_ARG_SENTINEL = "CFG026_TOOL_ARG_BYTE_LOCK"
ENCRYPTED_SENTINEL = "CFG026_ENCRYPTED_REASONING_LOCK"
SCHEMA_SENTINEL = "CFG026_SCHEMA_DESCRIPTION_LOCK"


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


def _kinds(detection: Any) -> set[str]:
    raw = _field(detection, "hit_kinds")
    return {str(kind).lower().replace("-", "_") for kind in raw}


def _has_kind(detection: Any, *candidates: str) -> bool:
    kinds = _kinds(detection)
    wanted = {item.lower().replace("-", "_") for item in candidates}
    return bool(kinds.intersection(wanted))


def _policy_body(result: Any, original: dict[str, Any]) -> dict[str, Any]:
    if result is None:
        return original
    body = _optional_field(result, "body", original)
    assert isinstance(body, dict)
    return body


def _policy_audit(result: Any) -> Any:
    if result is None:
        return None
    return _optional_field(result, "audit")


def _sanitize_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, tuple) and result:
        return str(result[0])
    return str(_field(result, "text"))


def _default_off_config() -> Any:
    return _load_text_watermark_config(None)


def _detect_config(**overrides: Any) -> Any:
    payload: dict[str, Any] = {
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
        "limits": {
            "max_text_bytes_per_direction": 1048576,
            "max_text_nodes_per_direction": 256,
            "max_reported_paths": 32,
            "max_reported_hits_per_path": 16,
        },
    }
    payload.update(overrides)
    return _load_text_watermark_config(payload)


def _sanitize_config(**overrides: Any) -> Any:
    payload: dict[str, Any] = {
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
    payload.update(overrides)
    return _load_text_watermark_config(payload)


def _responses_request_body() -> dict[str, Any]:
    return {
        "model": "gpt-5.4",
        "instructions": f"Follow the user. {PROMPT_SENTINEL}",
        "input": [
            {
                "type": "message",
                "role": "user",
                "id": "msg_visible",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Please inspect the pantry layout.",
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


def _chat_completions_request_body() -> dict[str, Any]:
    return {
        "model": "gpt-5.4",
        "messages": [
            {"role": "system", "content": "System visible text."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Chat user visible text."},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_chat_keep",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps(
                                {"cmd": "pwd", "note": TOOL_ARG_SENTINEL},
                                separators=(",", ":"),
                            ),
                        },
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "description": SCHEMA_SENTINEL,
                    },
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_default_config_mode_is_off_removal_disabled_audit_only_stream() -> None:
    cfg = _default_off_config()
    assert _field(cfg, "mode") == "off"
    removal = _field(cfg, "removal")
    assert _field(removal, "enabled") is False
    assert _field(removal, "stream_policy") == "audit_only"

    from litellm.proxy._types import ConfigGeneralSettings

    settings = ConfigGeneralSettings()
    typed = settings.openai_passthrough_text_watermark
    loaded = _load_text_watermark_config(typed)
    assert _field(loaded, "mode") == "off"
    assert _field(_field(loaded, "removal"), "enabled") is False
    assert _field(_field(loaded, "removal"), "stream_policy") == "audit_only"


@pytest.mark.parametrize("mode", ["sanitize", "enforce"])
def test_invalid_config_sanitize_or_enforce_without_removal_enabled_raises(
    mode: str,
) -> None:
    payload = {
        "mode": mode,
        "removal": {
            "enabled": False,
            "stream_policy": "buffer_response",
        },
    }
    with pytest.raises((ValueError, TypeError, ValidationError)) as info:
        _load_text_watermark_config(payload)
    message = str(info.value).lower()
    assert "removal" in message or "enabled" in message


def test_invalid_config_enforce_streamed_output_without_buffer_response_raises() -> None:
    payload = {
        "mode": "enforce",
        "removal": {
            "enabled": True,
            "stream_policy": "audit_only",
        },
    }
    with pytest.raises((ValueError, TypeError, ValidationError)) as info:
        _load_text_watermark_config(payload)
    message = str(info.value).lower()
    assert "buffer_response" in message or "stream" in message


def test_mode_off_policy_helper_is_noop_without_audit() -> None:
    body = {
        "instructions": f"hello{ZWSP}world",
        "input": "visible user text",
    }
    original = copy.deepcopy(body)
    result = _apply_watermark_policy(
        body=body,
        config=_default_off_config(),
        direction="request",
        endpoint="responses",
    )
    audit = _policy_audit(result)
    out_body = _policy_body(result, body)
    assert audit is None
    assert out_body == original
    assert ZWSP in out_body["instructions"]


# ---------------------------------------------------------------------------
# Visible text extraction
# ---------------------------------------------------------------------------


def test_extracts_visible_text_nodes_and_skips_protected_surfaces() -> None:
    responses = _responses_request_body()
    responses["input"] = "top-level string input"
    response_nodes = list(
        _extract_visible_text_nodes(
            responses,
            endpoint="responses",
            direction="request",
        )
    )
    response_texts = {_field(node, "text") for node in response_nodes}
    response_paths = {_field(node, "path") for node in response_nodes}
    joined_paths = " ".join(sorted(response_paths))

    assert any("Follow the user." in text for text in response_texts)
    assert "top-level string input" in response_texts
    assert any("instructions" in path for path in response_paths)
    assert any(path == "input" or path.endswith("input") for path in response_paths)

    message_body = _responses_request_body()
    message_nodes = list(
        _extract_visible_text_nodes(
            message_body,
            endpoint="responses",
            direction="request",
        )
    )
    message_texts = {_field(node, "text") for node in message_nodes}
    message_paths = " ".join(_field(node, "path") for node in message_nodes)
    assert "Please inspect the pantry layout." in message_texts
    assert "input_text" in message_paths or "content" in message_paths
    assert TOOL_ARG_SENTINEL not in message_texts
    assert ENCRYPTED_SENTINEL not in message_texts
    assert SCHEMA_SENTINEL not in message_texts
    assert "fc_keep_id" not in message_texts
    assert "rs_keep_id" not in message_texts
    assert "call_keep_id" not in message_texts
    assert "arguments" not in " ".join(_field(node, "path") for node in message_nodes)
    assert "encrypted" not in " ".join(
        _field(node, "path") for node in message_nodes
    ).lower()

    chat_nodes = list(
        _extract_visible_text_nodes(
            _chat_completions_request_body(),
            endpoint="chat_completions",
            direction="request",
        )
    )
    chat_texts = {_field(node, "text") for node in chat_nodes}
    chat_paths = " ".join(_field(node, "path") for node in chat_nodes)
    assert "System visible text." in chat_texts
    assert "Chat user visible text." in chat_texts
    assert "messages" in chat_paths
    assert TOOL_ARG_SENTINEL not in chat_texts
    assert SCHEMA_SENTINEL not in chat_texts
    assert "call_chat_keep" not in chat_texts
    assert "arguments" not in " ".join(_field(node, "path") for node in chat_nodes)
    assert joined_paths  # responses paths were collected


# ---------------------------------------------------------------------------
# Unicode detector
# ---------------------------------------------------------------------------


def test_conservative_unicode_carrier_detects_isolated_suspicious_codepoints() -> None:
    samples = {
        "zero_width_space": f"hello{ZWSP}world",
        "zwj_family": f"hello{ZWJ}world",
        "noncharacter": f"hello{NONCHARACTER}world",
        "exotic_space": f"hello{EXOTIC_EN_QUAD}world{EXOTIC_IDEOGRAPHIC_SPACE}end",
    }
    kind_aliases = {
        "zero_width_space": (
            "zero_width_space",
            "zwsp",
            "zero_width",
            "format_control",
        ),
        "zwj_family": ("zwj_family", "zwj", "isolated_zwj"),
        "noncharacter": ("noncharacter", "noncharacters"),
        "exotic_space": ("exotic_space", "exotic_spaces", "space"),
    }
    for label, text in samples.items():
        detection = _detect_unicode_carriers(text, policy="conservative")
        assert _field(detection, "signal_detected") is True
        assert _field(detection, "confirmed_watermark_detected") is False
        assert _field(detection, "vendor_attribution") == "unknown"
        assert _has_kind(detection, *kind_aliases[label]), (
            f"{label} should report a matching hit_kind, got {_kinds(detection)}"
        )


def test_conservative_policy_preserves_context_valid_emoji_zwj_sequences() -> None:
    for text in (EMOJI_FAMILY, EMOJI_RAINBOW_FLAG):
        detection = _detect_unicode_carriers(text, policy="conservative")
        assert _field(detection, "confirmed_watermark_detected") is False
        assert _field(detection, "vendor_attribution") == "unknown"
        assert _field(detection, "signal_detected") is False
        assert not _has_kind(
            detection,
            "zwj_family",
            "zwj",
            "isolated_zwj",
            "zero_width_space",
            "zwsp",
        )
        sanitized = _sanitize_unicode_carriers(text, policy="conservative")
        assert _sanitize_text(sanitized) == text


def test_conservative_sanitation_removes_isolated_zwsp_and_preserves_tool_args() -> None:
    body = _responses_request_body()
    body["instructions"] = f"Visible{ZWSP}instructions"
    body["input"][0]["content"][0]["text"] = f"user{ZWSP}question"
    original_args = body["input"][1]["arguments"]
    original_encrypted = body["input"][2]["encrypted_content"]
    original_schema = copy.deepcopy(body["tools"])
    original_ids = (
        body["input"][1]["id"],
        body["input"][1]["call_id"],
        body["input"][2]["id"],
    )

    result = _apply_watermark_policy(
        body=copy.deepcopy(body),
        config=_sanitize_config(),
        direction="request",
        endpoint="responses",
    )
    audit = _policy_audit(result)
    out_body = _policy_body(result, body)
    assert audit is not None
    assert ZWSP not in out_body["instructions"]
    assert ZWSP not in out_body["input"][0]["content"][0]["text"]
    assert out_body["input"][1]["arguments"] == original_args
    assert out_body["input"][2]["encrypted_content"] == original_encrypted
    assert out_body["tools"] == original_schema
    assert (
        out_body["input"][1]["id"],
        out_body["input"][1]["call_id"],
        out_body["input"][2]["id"],
    ) == original_ids

    redetect = _detect_unicode_carriers(
        out_body["instructions"] + out_body["input"][0]["content"][0]["text"],
        policy="conservative",
    )
    assert _field(redetect, "signal_detected") is False

    status = _field(audit, "status")
    transformation = _field(audit, "transformation")
    post_result = _optional_field(transformation, "result")
    post_status = _optional_field(transformation, "post_status")
    assert status in {"sanitized", "removed_verified"}
    assert post_result == "removed_verified" or post_status == "removed_verified"
    assert _field(audit, "confirmed_watermark_detected") is False
    assert _field(audit, "vendor_attribution") == "unknown"


def test_aggressive_policy_strips_variation_selectors_conservative_does_not() -> None:
    text = f"plain{VS16}text"
    conservative_detection = _detect_unicode_carriers(text, policy="conservative")
    assert _field(conservative_detection, "signal_detected") is True
    assert _has_kind(
        conservative_detection,
        "variation_selector",
        "variation_selectors",
        "vs",
    )
    conservative_cleaned = _sanitize_text(
        _sanitize_unicode_carriers(text, policy="conservative")
    )
    assert VS16 in conservative_cleaned

    aggressive_cleaned = _sanitize_text(
        _sanitize_unicode_carriers(text, policy="aggressive")
    )
    assert VS16 not in aggressive_cleaned
    assert "plain" in aggressive_cleaned
    assert "text" in aggressive_cleaned

    aggressive_cfg = _sanitize_config(
        unicode={
            "enabled": True,
            "policy": "aggressive",
            "normalize_spaces": True,
            "nfkc": False,
        }
    )
    assert _field(_field(aggressive_cfg, "unicode"), "policy") == "aggressive"


# ---------------------------------------------------------------------------
# Bounded audits
# ---------------------------------------------------------------------------


def test_audit_objects_are_bounded_without_raw_matches_or_full_prompt() -> None:
    contents = [
        f"{PROMPT_SENTINEL} node-{index} {ZWSP * 8} extra"
        for index in range(6)
    ]
    body = {
        "model": "gpt-5.4",
        "instructions": contents[0],
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            }
            for text in contents[1:]
        ],
    }
    cfg = _detect_config(
        limits={
            "max_text_bytes_per_direction": 1048576,
            "max_text_nodes_per_direction": 256,
            "max_reported_paths": 2,
            "max_reported_hits_per_path": 2,
        }
    )
    result = _apply_watermark_policy(
        body=body,
        config=cfg,
        direction="request",
        endpoint="responses",
    )
    audit = _policy_audit(result)
    assert audit is not None
    dumped = json.dumps(audit, default=str, ensure_ascii=False)
    assert ZWSP not in dumped
    assert "\\u200b" not in dumped.lower()
    assert PROMPT_SENTINEL not in dumped
    assert _field(audit, "truncated") is True
    paths = list(_field(audit, "paths"))
    assert len(paths) <= 2
    for path in paths:
        hit_count = _field(path, "hit_count")
        assert hit_count >= 1
        hits = _optional_field(path, "hits")
        if hits is not None:
            assert len(list(hits)) <= 2
        path_dump = json.dumps(path, default=str, ensure_ascii=False)
        assert ZWSP not in path_dump
        assert PROMPT_SENTINEL not in path_dump


# ---------------------------------------------------------------------------
# Statistical registry
# ---------------------------------------------------------------------------


def test_empty_or_disabled_statistical_registry_is_unsupported_and_skips_torch() -> None:
    forbidden_prefixes = ("torch", "transformers", "markllm")
    before = {
        name
        for name in sys.modules
        if name in forbidden_prefixes
        or any(name.startswith(f"{prefix}.") for prefix in forbidden_prefixes)
    }

    empty_cfg = _detect_config(statistical_detectors=[])
    empty_result = _evaluate_statistical_detectors(
        text="enough visible text to look like a generation " * 20,
        config=empty_cfg,
    )
    empty_status = _field(empty_result, "status")
    assert empty_status in {"unsupported", "inconclusive"}
    assert empty_status != "clean"

    disabled_cfg = _detect_config(
        statistical_detectors=[
            {
                "name": "internal_keyed_gumbel",
                "type": "keyed_gumbel",
                "enabled": False,
                "tokenizer": "internal-model-tokenizer",
                "key_secret_ref": "os.environ/INTERNAL_WATERMARK_KEY",
                "threshold": 2.33,
                "minimum_tokens": 64,
            }
        ]
    )
    disabled_result = _evaluate_statistical_detectors(
        text="enough visible text to look like a generation " * 20,
        config=disabled_cfg,
    )
    disabled_status = _field(disabled_result, "status")
    assert disabled_status in {"unsupported", "inconclusive"}
    assert disabled_status != "clean"

    after = {
        name
        for name in sys.modules
        if name in forbidden_prefixes
        or any(name.startswith(f"{prefix}.") for prefix in forbidden_prefixes)
    }
    assert after == before


# ---------------------------------------------------------------------------
# Session-history allowlist
# ---------------------------------------------------------------------------


def test_session_history_metadata_keys_include_watermark_audits() -> None:
    from litellm.integrations.aawm_agent_identity.constants import (
        _AAWM_SESSION_HISTORY_METADATA_KEYS,
    )

    assert "watermark_input_audit" in _AAWM_SESSION_HISTORY_METADATA_KEYS
    assert "watermark_output_audit" in _AAWM_SESSION_HISTORY_METADATA_KEYS
