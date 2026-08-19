"""Streaming iterator for Cursor Agent CLI interaction_update frames."""

from __future__ import annotations

from typing import Union

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.utils import GenericStreamingChunk, ModelResponseStream

from ..common_utils import extract_text_from_agent_payload


class CursorAgentModelResponseIterator(BaseModelResponseIterator):
    def chunk_parser(
        self, chunk: dict
    ) -> Union[GenericStreamingChunk, ModelResponseStream]:
        text, ended = extract_text_from_agent_payload(chunk)
        return GenericStreamingChunk(
            text=text,
            is_finished=ended,
            finish_reason="stop" if ended else "",
            usage=None,
            index=0,
            tool_use=None,
        )
