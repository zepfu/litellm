"""Google Code Assist shaping implementation slice.

This module owns provider algorithms. Route-layer dependencies are rebound by
the compatibility facade before entry so existing monkeypatch contracts remain
live without suppressing undefined-name checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Optional, cast

import httpx
from fastapi import HTTPException, Response

from litellm.types.llms.openai import (
    AllMessageValues,
)


@dataclass(frozen=True)
class Runtime:
    _build_anthropic_response_from_completion_adapter_response: Any
    _decode_http_response_body: Any
    _normalize_google_completion_adapter_model_name: Any


_RUNTIME: Optional[Runtime] = None


def bind_runtime(namespace: Mapping[str, object]) -> None:
    global _RUNTIME
    _RUNTIME = Runtime(
        _build_anthropic_response_from_completion_adapter_response=namespace[
            "_build_anthropic_response_from_completion_adapter_response"
        ],
        _decode_http_response_body=namespace["_decode_http_response_body"],
        _normalize_google_completion_adapter_model_name=namespace["_normalize_google_completion_adapter_model_name"],
    )


def _runtime() -> Runtime:
    if _RUNTIME is None:
        raise RuntimeError("Google shaping runtime is not bound")
    return _RUNTIME


def _extract_google_code_assist_text_metrics(content_block: Any) -> tuple[int, int]:
    part_count = 0
    char_count = 0
    if not isinstance(content_block, dict):
        return part_count, char_count
    parts = content_block.get("parts")
    if not isinstance(parts, list):
        return part_count, char_count
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_count += 1
        text = part.get("text")
        if isinstance(text, str):
            char_count += len(text)
    return part_count, char_count


def _summarize_google_code_assist_content_preview_entry(
    content_entry: dict[str, Any],
) -> dict[str, Any]:
    role = content_entry.get("role")
    parts = content_entry.get("parts")
    part_kinds = []
    text_preview = None
    preview_parts, preview_chars = _extract_google_code_assist_text_metrics(content_entry)
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            keys = [key for key in ("text", "functionCall", "functionResponse", "thought") if key in part]
            if keys:
                part_kinds.extend(keys)
            text_value = part.get("text")
            if text_preview is None and isinstance(text_value, str):
                text_preview = text_value[:120].replace("\n", "\\n")
            function_response = part.get("functionResponse")
            if isinstance(function_response, dict):
                response_payload = function_response.get("response")
                if isinstance(response_payload, dict):
                    response_keys = sorted(response_payload.keys())
                    part_kinds.append(f"functionResponseKeys:{','.join(response_keys)}")
                    content_value = response_payload.get("content")
                    if text_preview is None and isinstance(content_value, str):
                        text_preview = content_value[:120].replace("\n", "\\n")
    return {
        "role": role,
        "part_count": preview_parts,
        "text_chars": preview_chars,
        "part_kinds": part_kinds,
        "text_preview": text_preview,
    }


def _summarize_google_code_assist_request_contents_shape(
    request_block: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    contents = request_block.get("contents")
    if not isinstance(contents, list):
        return

    summary["contents_count"] = len(contents)
    content_part_count = 0
    content_text_chars = 0
    text_entry_count = 0
    preview_entries = []
    for content_entry in contents:
        parts, chars = _extract_google_code_assist_text_metrics(content_entry)
        content_part_count += parts
        content_text_chars += chars
        if chars > 0:
            text_entry_count += 1
    for content_entry in contents[-4:]:
        if isinstance(content_entry, dict):
            preview_entries.append(_summarize_google_code_assist_content_preview_entry(content_entry))
    summary["contents_part_count"] = content_part_count
    summary["contents_text_chars"] = content_text_chars
    summary["contents_text_entry_count"] = text_entry_count
    summary["contents_tail_preview"] = preview_entries


def _summarize_google_code_assist_generation_config_shape(
    request_block: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    generation_config = request_block.get("generationConfig")
    if not isinstance(generation_config, dict):
        return

    summary["generation_config_keys"] = sorted(generation_config.keys())
    generation_config_summary = {}
    for key in ("max_output_tokens", "temperature", "top_p", "candidate_count"):
        if key in generation_config:
            generation_config_summary[key] = generation_config.get(key)
    thinking_config = generation_config.get("thinkingConfig")
    if isinstance(thinking_config, dict):
        generation_config_summary["thinking_config_keys"] = sorted(thinking_config.keys())
        if "thinkingBudgetTokens" in thinking_config:
            generation_config_summary["thinking_budget_tokens"] = thinking_config.get("thinkingBudgetTokens")
    if generation_config_summary:
        summary["generation_config_values"] = generation_config_summary


def _extract_google_code_assist_function_names(request_block: Any) -> list[str]:
    request_tools = request_block.get("tools") if isinstance(request_block, dict) else None
    function_names: list[str] = []
    if not isinstance(request_tools, list):
        return function_names

    for tool_entry in request_tools:
        if not isinstance(tool_entry, dict):
            continue
        decls = tool_entry.get("functionDeclarations") or tool_entry.get("function_declarations")
        if not isinstance(decls, list):
            continue
        for declaration in decls:
            if not isinstance(declaration, dict):
                continue
            name = declaration.get("name")
            if isinstance(name, str):
                function_names.append(name)
    return function_names


def _summarize_google_code_assist_request_shape(payload: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if not isinstance(payload, dict):
        summary["payload_type"] = type(payload).__name__
        return summary

    summary["top_level_keys"] = sorted(payload.keys())
    for key in ("model", "project", "user_prompt_id", "session_id"):
        if key in payload:
            summary[key] = payload.get(key)

    request_block = payload.get("request") if isinstance(payload.get("request"), dict) else payload
    if isinstance(request_block, dict):
        summary["request_keys"] = sorted(request_block.keys())
        _summarize_google_code_assist_request_contents_shape(request_block, summary)
        tools = request_block.get("tools")
        if isinstance(tools, list):
            summary["tools_count"] = len(tools)
            function_declaration_count = 0
            for tool_entry in tools:
                if isinstance(tool_entry, dict):
                    decls = tool_entry.get("functionDeclarations") or tool_entry.get("function_declarations")
                    if isinstance(decls, list):
                        function_declaration_count += len(decls)
            summary["function_declaration_count"] = function_declaration_count
        session_id = request_block.get("session_id")
        if isinstance(session_id, str) and session_id:
            summary["session_id_hash"] = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:8]
        _summarize_google_code_assist_generation_config_shape(request_block, summary)
        tool_config = request_block.get("toolConfig")
        if isinstance(tool_config, dict):
            summary["tool_config_keys"] = sorted(tool_config.keys())
        system_instruction = request_block.get("systemInstruction")
        if isinstance(system_instruction, dict):
            summary["has_system_instruction"] = True
            system_parts, system_chars = _extract_google_code_assist_text_metrics(system_instruction)
            summary["system_instruction_part_count"] = system_parts
            summary["system_instruction_text_chars"] = system_chars
    return summary


def _unwrap_google_code_assist_response_payload(
    payload: Any,
) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    response_payload = payload.get("response")
    if not isinstance(response_payload, dict):
        return None
    unwrapped = dict(response_payload)
    trace_id = payload.get("traceId")
    if isinstance(trace_id, str) and trace_id and "responseId" not in unwrapped:
        unwrapped["responseId"] = trace_id
    return unwrapped


async def _translate_google_code_assist_response_to_anthropic(
    *,
    response: Response,
    adapter_model: str,
    tool_name_mapping: dict[str, str],
    completion_messages: list[dict[str, Any]],
    gemini_optional_params: dict[str, Any],
    litellm_params: dict[str, Any],
    logging_obj: Any,
) -> Response:
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        AnthropicAdapter,
    )
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    from litellm.main import _get_encoding
    from litellm.utils import ModelResponse

    try:
        outer_payload = json.loads(_runtime()._decode_http_response_body(response.body))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Google Code Assist adapter returned invalid JSON: {exc}",
        ) from exc

    unwrapped_payload = _unwrap_google_code_assist_response_payload(outer_payload)
    if unwrapped_payload is None:
        raise HTTPException(
            status_code=502,
            detail="Google Code Assist adapter response did not contain a `response` payload.",
        )

    raw_response = httpx.Response(
        status_code=response.status_code,
        headers=dict(response.headers),
        content=json.dumps(unwrapped_payload).encode("utf-8"),
    )
    model_response = VertexGeminiConfig().transform_response(
        model=_runtime()._normalize_google_completion_adapter_model_name(adapter_model),
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        request_data=unwrapped_payload,
        messages=cast(list[AllMessageValues], completion_messages),
        optional_params=gemini_optional_params,
        litellm_params=litellm_params,
        encoding=_get_encoding(),
        api_key="",
    )
    anthropic_response = AnthropicAdapter().translate_completion_output_params(
        model_response,
        tool_name_mapping=tool_name_mapping,
    )
    return _runtime()._build_anthropic_response_from_completion_adapter_response(anthropic_response)
