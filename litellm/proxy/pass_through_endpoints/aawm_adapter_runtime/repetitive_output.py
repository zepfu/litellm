"""CFG-025: config-driven repetitive visible-text guard.

Detects looping user-visible Responses output and aborts the OpenAI
passthrough stream with ``response.failed`` / ``aawm_repetitive_output_loop``.
Thresholds come from named policies; this module has no model-name branches.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping, Optional

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.sse import (
    _iter_sse_event_blocks_with_separator,
    _mapping_or_attr_get,
    _responses_event_text_key,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.output_guard_config import (
    OutputGuardPolicy,
    OutputGuardRequestContext,
    resolve_output_guard_policy,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

REPETITIVE_OUTPUT_FAILURE_KIND = "repetitive_output_loop"
REPETITIVE_OUTPUT_ERROR_CODE = "aawm_repetitive_output_loop"
REPETITIVE_OUTPUT_ERROR_TYPE = "proxy_stream_terminal_error"
REPETITIVE_OUTPUT_MESSAGE = (
    "Repetitive visible output loop detected; stream aborted without retry."
)

_VISIBLE_DELTA_TYPE = "response.output_text.delta"
_VISIBLE_DONE_TYPE = "response.output_text.done"
_MESSAGE_ITEM_TYPE = "message"
_VISIBLE_PART_TYPES = frozenset({"output_text", "text"})
_TOKEN_KEEP = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_./")
WRAPPED_STREAM_ATTR = "_aawm_repetitive_output_guard_wrapped"
OUTPUT_GUARD_CONTEXT_ATTR = "_aawm_output_guard_request_context"


def _event_type(event: Any) -> str:
    value = _mapping_or_attr_get(event, "type")
    return value if isinstance(value, str) else ""


def extract_visible_text_from_responses_event(event: Any) -> Optional[str]:
    """Return user-visible output_text payload, excluding reasoning/tool args."""
    event_type = _event_type(event)
    if event_type == _VISIBLE_DELTA_TYPE:
        delta = _mapping_or_attr_get(event, "delta")
        return delta if isinstance(delta, str) and delta else None
    if event_type == _VISIBLE_DONE_TYPE:
        text = _mapping_or_attr_get(event, "text")
        return text if isinstance(text, str) and text else None
    return None


class VisibleOutputTextExtractor:
    """Accumulate visible Responses text with the stream_collect delta/done rule."""

    def __init__(self) -> None:
        self._delta_keys_seen: set[str] = set()

    def consume_event(self, event: Any) -> Optional[str]:
        event_type = _event_type(event)
        if event_type == _VISIBLE_DELTA_TYPE:
            text = extract_visible_text_from_responses_event(event)
            if text:
                self._delta_keys_seen.add(_responses_event_text_key(event))
            return text
        if event_type == _VISIBLE_DONE_TYPE:
            text_key = _responses_event_text_key(event)
            if text_key in self._delta_keys_seen:
                return None
            text = extract_visible_text_from_responses_event(event)
            if text:
                self._delta_keys_seen.add(text_key)
            return text
        return None


def extract_visible_output_text_from_response_body(body: Mapping[str, Any]) -> str:
    """Concatenate assistant message output_text/text parts from a JSON body."""
    output = body.get("output") if isinstance(body, Mapping) else None
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != _MESSAGE_ITEM_TYPE:
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in _VISIBLE_PART_TYPES:
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts)


def _normalize_visible_text_with_remainder(
    text: str,
    *,
    pending: str = "",
    flush: bool = True,
) -> tuple[list[str], str]:
    """Tokenize visible text, optionally holding a split-token remainder.

    SSE ``output_text.delta`` chunks can bisect a word. Concatenate ``pending``
    from the previous chunk and keep the trailing keep-character token until a
    later boundary (or ``flush=True``) so n-grams match the assembled stream.
    """
    combined = f"{pending}{text}" if pending else text
    if not combined:
        return [], ""
    normalized = unicodedata.normalize("NFKC", combined).lower()
    chars: list[str] = []
    last_keep = False
    for char in normalized:
        if char.isspace():
            chars.append(" ")
            last_keep = False
        elif char in _TOKEN_KEEP:
            chars.append(char)
            last_keep = True
        else:
            chars.append(" ")
            last_keep = False
    collapsed = " ".join("".join(chars).split())
    if not collapsed:
        return [], ""
    words = collapsed.split()
    if flush or not last_keep:
        return words, ""
    return words[:-1], words[-1]


def normalize_visible_text(text: str) -> list[str]:
    """NFKC, lowercase, strip noise punctuation, collapse whitespace."""
    words, _remainder = _normalize_visible_text_with_remainder(
        text, pending="", flush=True
    )
    return words


def _token_edit_distance(left: tuple[str, ...], right: tuple[str, ...], cap: int) -> int:
    if left == right:
        return 0
    if cap < 0:
        return 0
    if abs(len(left) - len(right)) > cap:
        return cap + 1
    prev = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        row_min = i
        for j, right_token in enumerate(right, start=1):
            cost = 0 if left_token == right_token else 1
            current.append(
                min(
                    prev[j] + 1,
                    current[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
            if current[-1] < row_min:
                row_min = current[-1]
        if row_min > cap:
            return cap + 1
        prev = current
    return prev[-1]


def _ngram_fingerprint(ngram: tuple[str, ...]) -> str:
    payload = "\x1f".join(ngram).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_fingerprint(fingerprint: str) -> str:
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def _unique_trigram_novelty(words: list[str], growth_words: int) -> float:
    if growth_words <= 0:
        return 1.0
    window = words[-growth_words:]
    if len(window) < 3:
        return 1.0
    trigrams = [tuple(window[index : index + 3]) for index in range(len(window) - 2)]
    if not trigrams:
        return 1.0
    return len(set(trigrams)) / float(len(trigrams))


def _best_repeated_ngram(
    words: list[str],
    *,
    min_ngram: int,
    min_repeats: int,
    max_edit_distance: int,
) -> Optional[tuple[tuple[str, ...], int, int]]:
    if min_ngram <= 0 or len(words) < min_ngram:
        return None
    starts_by_ngram: dict[tuple[str, ...], list[int]] = {}
    for index in range(len(words) - min_ngram + 1):
        ngram = tuple(words[index : index + min_ngram])
        starts_by_ngram.setdefault(ngram, []).append(index)
    if not starts_by_ngram:
        return None

    assigned: set[tuple[str, ...]] = set()
    best: Optional[tuple[tuple[str, ...], int, int]] = None
    ordered = sorted(
        starts_by_ngram.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    for ngram, starts in ordered:
        if ngram in assigned:
            continue
        cluster_starts = list(starts)
        assigned.add(ngram)
        if max_edit_distance > 0:
            for other, other_starts in starts_by_ngram.items():
                if other in assigned:
                    continue
                if _token_edit_distance(ngram, other, max_edit_distance) <= max_edit_distance:
                    cluster_starts.extend(other_starts)
                    assigned.add(other)
        covered: set[int] = set()
        for start in cluster_starts:
            covered.update(range(start, start + min_ngram))
        candidate = (ngram, len(cluster_starts), len(covered))
        if best is None or candidate[1] > best[1] or (
            candidate[1] == best[1] and candidate[2] > best[2]
        ):
            best = candidate
    if best is None or best[1] < min_repeats:
        return None
    return best


@dataclass(frozen=True)
class VisibleTextRepetitionMatch:
    policy_name: str
    word_count: int
    repeat_count: int
    coverage: float
    novelty: float
    triggering_ngram_fingerprint: str
    window_size: int
    ngram_length: int

    def diagnostics(self) -> dict[str, Any]:
        return {
            "policy": self.policy_name,
            "window_size": self.window_size,
            "ngram_length": self.ngram_length,
            "repeat_count": self.repeat_count,
            "coverage": self.coverage,
            "novelty": self.novelty,
            "word_count": self.word_count,
            "triggering_ngram_hash": _hash_fingerprint(self.triggering_ngram_fingerprint),
            "failure_kind": REPETITIVE_OUTPUT_FAILURE_KIND,
        }


class VisibleTextRepetitionDetector:
    """Rolling-window scorer over normalized visible words."""

    def __init__(self, policy: OutputGuardPolicy) -> None:
        self.policy = policy
        self._window: deque[str] = deque(maxlen=policy.window_words)
        self._total_words = 0
        self._pending = ""
        self._match: Optional[VisibleTextRepetitionMatch] = None
        self._extractor = VisibleOutputTextExtractor()

    def normalized_words(self) -> list[str]:
        return list(self._window)

    def feed(
        self, text: str, *, flush: bool = True
    ) -> Optional[VisibleTextRepetitionMatch]:
        if self._match is not None:
            return self._match
        words, self._pending = _normalize_visible_text_with_remainder(
            text, pending=self._pending, flush=flush
        )
        for word in words:
            self._window.append(word)
            self._total_words += 1
        self._match = self._score()
        return self._match

    def feed_event(self, event: Any) -> Optional[VisibleTextRepetitionMatch]:
        if self._match is not None:
            return self._match
        text = self._extractor.consume_event(event)
        if not text:
            return None
        return self.feed(text, flush=False)

    def _score(self) -> Optional[VisibleTextRepetitionMatch]:
        policy = self.policy
        if self._total_words < policy.min_words:
            return None
        words = list(self._window)
        if len(words) < policy.min_ngram:
            return None
        best = _best_repeated_ngram(
            words,
            min_ngram=policy.min_ngram,
            min_repeats=policy.min_repeats,
            max_edit_distance=policy.max_ngram_edit_distance,
        )
        if best is None:
            return None
        ngram, repeat_count, covered_count = best
        coverage = covered_count / float(len(words)) if words else 0.0
        if coverage < policy.min_coverage:
            return None
        novelty = _unique_trigram_novelty(words, policy.growth_words)
        if novelty > policy.max_novelty:
            return None
        return VisibleTextRepetitionMatch(
            policy_name=policy.name,
            word_count=self._total_words,
            repeat_count=repeat_count,
            coverage=coverage,
            novelty=novelty,
            triggering_ngram_fingerprint=_ngram_fingerprint(ngram),
            window_size=len(words),
            ngram_length=policy.min_ngram,
        )


def _parse_sse_event_block(event_block: str) -> list[Any]:
    events: list[Any] = []
    for line in event_block.splitlines():
        parsed = BaseModelResponseIterator._string_to_dict_parser(line)
        if parsed is not None:
            events.append(parsed)
    return events


def _encode_sse_block(event_block: str, had_separator: bool) -> bytes:
    text = event_block if event_block.endswith("\n\n") or not had_separator else f"{event_block}\n\n"
    if had_separator and not text.endswith("\n\n"):
        text = f"{event_block}\n\n"
    return text.encode("utf-8")


def _as_bytes(chunk: Any) -> bytes:
    if isinstance(chunk, bytes):
        return chunk
    if isinstance(chunk, bytearray):
        return bytes(chunk)
    return str(chunk).encode("utf-8")


async def _close_upstream_response(upstream_response: Any) -> None:
    if upstream_response is None:
        return
    close_fn = getattr(upstream_response, "aclose", None)
    if not callable(close_fn):
        close_fn = getattr(upstream_response, "close", None)
    if not callable(close_fn):
        return
    result = close_fn()
    if inspect.isawaitable(result):
        await result


def _terminal_metadata(
    *,
    match: VisibleTextRepetitionMatch,
    request_context: OutputGuardRequestContext,
    termination_action: str,
    visible_delta_forwarded: bool,
) -> dict[str, Any]:
    diagnostics = match.diagnostics()
    return {
        **diagnostics,
        "policy": match.policy_name,
        "failure_kind": REPETITIVE_OUTPUT_FAILURE_KIND,
        "termination_action": termination_action,
        "visible_delta_forwarded": visible_delta_forwarded,
        "provider": request_context.custom_llm_provider,
        "route_family": request_context.route_family,
        "resolved_model": request_context.resolved_model,
        "ingress": request_context.ingress_path,
    }


def _encode_response_failed_sse(payload: Mapping[str, Any]) -> list[bytes]:
    return [
        (
            "event: response.failed\ndata: "
            + json.dumps(payload, separators=(",", ":"))
            + "\n\n"
        ).encode("utf-8"),
        b"data: [DONE]\n\n",
    ]


def _local_repetitive_output_failed_payload(
    *,
    match: VisibleTextRepetitionMatch,
    request_context: OutputGuardRequestContext,
    visible_delta_forwarded: bool,
) -> dict[str, Any]:
    return {
        "type": "response.failed",
        "response": {
            "object": "response",
            "status": "failed",
            "error": {
                "type": REPETITIVE_OUTPUT_ERROR_TYPE,
                "code": REPETITIVE_OUTPUT_ERROR_CODE,
                "message": REPETITIVE_OUTPUT_MESSAGE,
                "param": None,
            },
            "metadata": _terminal_metadata(
                match=match,
                request_context=request_context,
                termination_action="response.failed",
                visible_delta_forwarded=visible_delta_forwarded,
            ),
        },
    }


def _chunks_include_response_failed(chunks: list[bytes]) -> bool:
    return any(b"response.failed" in chunk for chunk in chunks)


def _build_repetitive_output_failed_chunks(
    *,
    match: VisibleTextRepetitionMatch,
    request_context: OutputGuardRequestContext,
    visible_delta_forwarded: bool,
) -> list[bytes]:
    local_payload = _local_repetitive_output_failed_payload(
        match=match,
        request_context=request_context,
        visible_delta_forwarded=visible_delta_forwarded,
    )
    try:
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )

        failure_context = {
            "failure_kind": REPETITIVE_OUTPUT_FAILURE_KIND,
            "error_code": REPETITIVE_OUTPUT_ERROR_CODE,
            "error_message": REPETITIVE_OUTPUT_MESSAGE,
            "responses_api_terminal": True,
            "model": request_context.resolved_model,
            "route_family": request_context.route_family,
            "terminal_metadata": local_payload["response"]["metadata"],
        }
        chunks = PassThroughStreamingHandler._build_post_first_byte_terminal_stream_chunks(
            endpoint_type=EndpointType.OPENAI,
            url_route=request_context.ingress_path or "/openai_passthrough/v1/responses",
            custom_llm_provider=request_context.custom_llm_provider,
            failure_context=failure_context,
            exc=None,
        )
    except Exception:
        chunks = []
    if chunks and _chunks_include_response_failed(chunks):
        return chunks
    return _encode_response_failed_sse(local_payload)


async def wrap_responses_sse_with_repetitive_output_guard(
    body_iterator: Any,
    *,
    policy: OutputGuardPolicy,
    request_context: OutputGuardRequestContext,
    upstream_response: Any = None,
    retry_fn: Any = None,
    failover_fn: Any = None,
) -> AsyncIterator[bytes]:
    """Forward Responses SSE until visible text loops, then fail closed."""
    del retry_fn, failover_fn
    detector = VisibleTextRepetitionDetector(policy)
    extractor = VisibleOutputTextExtractor()
    yielded_visible = False
    aborted = False
    closed_upstream = False
    sse_iter = _iter_sse_event_blocks_with_separator(body_iterator)

    async def _abort_upstream() -> None:
        nonlocal closed_upstream
        if closed_upstream:
            return
        closed_upstream = True
        await _close_upstream_response(upstream_response)

    try:
        async for event_block, had_separator in sse_iter:
            raw_chunk = _encode_sse_block(event_block, had_separator)
            events = _parse_sse_event_block(event_block)
            match: Optional[VisibleTextRepetitionMatch] = None
            block_has_visible = False
            for event in events:
                visible = extractor.consume_event(event)
                if not visible:
                    continue
                block_has_visible = True
                match = detector.feed(visible, flush=False)
                if match is not None:
                    break
            if match is not None:
                aborted = True
                visible_delta_forwarded = yielded_visible
                await _abort_upstream()
                await _close_upstream_response(sse_iter)
                await _close_upstream_response(body_iterator)
                if visible_delta_forwarded:
                    verbose_proxy_logger.warning(
                        "CFG-025 aborted repetitive Responses stream policy=%s "
                        "diagnostics=%s",
                        policy.name,
                        json.dumps(
                            _terminal_metadata(
                                match=match,
                                request_context=request_context,
                                termination_action="response.failed",
                                visible_delta_forwarded=True,
                            ),
                            separators=(",", ":"),
                        ),
                    )
                for chunk in _build_repetitive_output_failed_chunks(
                    match=match,
                    request_context=request_context,
                    visible_delta_forwarded=visible_delta_forwarded,
                ):
                    yield chunk
                return
            yield raw_chunk
            if block_has_visible:
                yielded_visible = True
    finally:
        await _close_upstream_response(sse_iter)
        if aborted:
            await _abort_upstream()


def maybe_wrap_passthrough_responses_stream(
    body_iterator: Any,
    *,
    request_context: OutputGuardRequestContext,
    upstream_response: Any = None,
    retry_fn: Any = None,
    failover_fn: Any = None,
    policy: Optional[OutputGuardPolicy] = None,
) -> AsyncIterator[Any]:
    """Install the SSE guard only when the named selector matches.

    Selector-off returns ``body_iterator`` unchanged so non-matching routes
    (direct ``/grok``, generic OpenAI, chat completions) keep their original
    stream object.
    """
    selected = policy or resolve_output_guard_policy(request_context)
    if selected is None:
        return body_iterator
    if getattr(body_iterator, WRAPPED_STREAM_ATTR, False):
        return body_iterator
    wrapped = wrap_responses_sse_with_repetitive_output_guard(
        body_iterator,
        policy=selected,
        request_context=request_context,
        upstream_response=upstream_response,
        retry_fn=retry_fn,
        failover_fn=failover_fn,
    )
    try:
        setattr(wrapped, WRAPPED_STREAM_ATTR, True)
    except Exception:
        pass
    return wrapped


def bind_output_guard_to_streaming_response(
    response: Any,
    *,
    request_context: OutputGuardRequestContext,
    wrapped: Optional[bool] = None,
) -> Any:
    """Attach selector context / wrap flag so reconstructed SSE can inherit the guard."""
    iterator = getattr(response, "body_iterator", None)
    if wrapped is None:
        wrapped = bool(
            getattr(response, WRAPPED_STREAM_ATTR, False)
            or getattr(iterator, WRAPPED_STREAM_ATTR, False)
        )
    try:
        setattr(response, OUTPUT_GUARD_CONTEXT_ATTR, request_context)
        if wrapped:
            setattr(response, WRAPPED_STREAM_ATTR, True)
    except Exception:
        pass
    return response


def inherit_or_wrap_passthrough_streaming_response(
    response: Any,
    *,
    source_response: Any = None,
    request_context: Optional[OutputGuardRequestContext] = None,
    upstream_response: Any = None,
) -> Any:
    """Wrap a reconstructed StreamingResponse unless the source stream is already guarded."""
    from fastapi.responses import StreamingResponse

    if not isinstance(response, StreamingResponse):
        return response
    context = request_context or getattr(response, OUTPUT_GUARD_CONTEXT_ATTR, None)
    if context is None and source_response is not None:
        context = getattr(source_response, OUTPUT_GUARD_CONTEXT_ATTR, None)
    if context is None:
        return response
    source_wrapped = bool(
        source_response is not None
        and (
            getattr(source_response, WRAPPED_STREAM_ATTR, False)
            or getattr(
                getattr(source_response, "body_iterator", None),
                WRAPPED_STREAM_ATTR,
                False,
            )
        )
    )
    if source_wrapped or getattr(response, WRAPPED_STREAM_ATTR, False):
        return bind_output_guard_to_streaming_response(
            response,
            request_context=context,
            wrapped=True,
        )
    wrapped_iter = maybe_wrap_passthrough_responses_stream(
        response.body_iterator,
        request_context=context,
        upstream_response=upstream_response,
    )
    if wrapped_iter is response.body_iterator:
        return bind_output_guard_to_streaming_response(
            response,
            request_context=context,
            wrapped=False,
        )
    guarded = StreamingResponse(
        wrapped_iter,
        headers=dict(response.headers),
        status_code=response.status_code,
        media_type=response.media_type or "text/event-stream",
    )
    return bind_output_guard_to_streaming_response(
        guarded,
        request_context=context,
        wrapped=True,
    )


def is_repetitive_output_loop_failure(body: Any) -> bool:
    """Return whether a Responses payload was aborted by this guard."""
    if not isinstance(body, Mapping):
        return False
    error = body.get("error")
    if isinstance(error, Mapping) and error.get("code") == REPETITIVE_OUTPUT_ERROR_CODE:
        return True
    nested = body.get("response")
    if isinstance(nested, Mapping):
        nested_error = nested.get("error")
        if (
            isinstance(nested_error, Mapping)
            and nested_error.get("code") == REPETITIVE_OUTPUT_ERROR_CODE
        ):
            return True
        nested_metadata = nested.get("metadata")
        if (
            isinstance(nested_metadata, Mapping)
            and nested_metadata.get("failure_kind") == REPETITIVE_OUTPUT_FAILURE_KIND
        ):
            return True
    metadata = body.get("metadata")
    return (
        isinstance(metadata, Mapping)
        and metadata.get("failure_kind") == REPETITIVE_OUTPUT_FAILURE_KIND
    )


def reject_nonstream_responses_body_if_repetitive(
    body: Mapping[str, Any],
    *,
    policy: OutputGuardPolicy,
    request_context: OutputGuardRequestContext,
) -> Mapping[str, Any]:
    """Raise HTTP 502 when a completed JSON Responses body is looped text."""
    text = extract_visible_output_text_from_response_body(body)
    match = VisibleTextRepetitionDetector(policy).feed(text)
    if match is None:
        return body
    metadata = _terminal_metadata(
        match=match,
        request_context=request_context,
        termination_action="http_502",
        visible_delta_forwarded=False,
    )
    verbose_proxy_logger.warning(
        "CFG-025 rejected repetitive non-stream Responses body policy=%s diagnostics=%s",
        policy.name,
        json.dumps(metadata, separators=(",", ":")),
    )
    raise HTTPException(
        status_code=502,
        detail={
            "error": {
                "type": REPETITIVE_OUTPUT_ERROR_TYPE,
                "code": REPETITIVE_OUTPUT_ERROR_CODE,
                "message": REPETITIVE_OUTPUT_MESSAGE,
                "param": None,
            },
            "metadata": metadata,
        },
    )


def maybe_reject_passthrough_responses_body(
    body: Any,
    *,
    request_context: OutputGuardRequestContext,
    policy: Optional[OutputGuardPolicy] = None,
) -> Any:
    """No-op unless the selector matches and the JSON body is looped."""
    if not isinstance(body, Mapping):
        return body
    selected = policy or resolve_output_guard_policy(request_context)
    if selected is None:
        return body
    return reject_nonstream_responses_body_if_repetitive(
        body,
        policy=selected,
        request_context=request_context,
    )


__all__ = [
    "OUTPUT_GUARD_CONTEXT_ATTR",
    "REPETITIVE_OUTPUT_ERROR_CODE",
    "REPETITIVE_OUTPUT_FAILURE_KIND",
    "WRAPPED_STREAM_ATTR",
    "VisibleOutputTextExtractor",
    "VisibleTextRepetitionDetector",
    "VisibleTextRepetitionMatch",
    "bind_output_guard_to_streaming_response",
    "extract_visible_output_text_from_response_body",
    "extract_visible_text_from_responses_event",
    "inherit_or_wrap_passthrough_streaming_response",
    "is_repetitive_output_loop_failure",
    "maybe_reject_passthrough_responses_body",
    "maybe_wrap_passthrough_responses_stream",
    "normalize_visible_text",
    "reject_nonstream_responses_body_if_repetitive",
    "wrap_responses_sse_with_repetitive_output_guard",
]
