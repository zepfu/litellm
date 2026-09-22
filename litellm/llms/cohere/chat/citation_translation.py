"""Translate Cohere V2 citations into OpenAI annotations and provider metadata.

Document sources with a real HTTP(S) URL and title become ``url_citation``
annotations. Every retained source, including tool outputs, keeps its source
id and character offsets in provider metadata. Tool-output bodies are omitted.
No URL is synthesized.
"""

from typing import Any, Dict, List, Optional, Tuple

from litellm.types.llms.openai import (
    ChatCompletionAnnotation,
    ChatCompletionAnnotationURLCitation,
)

_DOCUMENT_SOURCE = "document"
_TOOL_SOURCE = "tool"
_ALLOWED_SOURCE_TYPES = frozenset({_DOCUMENT_SOURCE, _TOOL_SOURCE})


def translate_cohere_v2_citations(
    citations: Any,
) -> Tuple[List[ChatCompletionAnnotation], Dict[str, Any]]:
    """Return ``(annotations, provider_specific_fields)``.

    ``provider_specific_fields`` is empty when no citation identity can be kept.
    The returned metadata never includes tool-output bodies, document snippets,
    or generated citation text.
    """
    annotations: List[ChatCompletionAnnotation] = []
    translated_citations: List[Dict[str, Any]] = []

    for citation in _citation_records(citations):
        translated = _translate_citation(citation, annotations)
        if translated is not None:
            translated_citations.append(translated)

    if not translated_citations:
        return annotations, {}
    return annotations, {"citations": translated_citations}


def cohere_citation_provider_fields(citations: Any) -> Optional[Dict[str, Any]]:
    """Map citation payloads to safe provider metadata, dropping tool bodies."""
    _annotations, provider_fields = translate_cohere_v2_citations(citations)
    if not provider_fields:
        return None
    return provider_fields


def _citation_records(citations: Any) -> List[dict]:
    if isinstance(citations, dict):
        citations = [citations]
    if not isinstance(citations, list):
        return []
    return [citation for citation in citations if isinstance(citation, dict)]


def _translate_citation(
    citation: dict,
    annotations: List[ChatCompletionAnnotation],
) -> Optional[Dict[str, Any]]:
    sources = citation.get("sources")
    if not isinstance(sources, list):
        return None

    translated_sources: List[Dict[str, Any]] = []
    for source in sources:
        translated_source = _translate_source(source)
        if translated_source is None:
            continue
        translated_sources.append(translated_source)
        annotation = _document_url_annotation(
            citation,
            translated_source,
        )
        if annotation is not None:
            annotations.append(annotation)

    if not translated_sources:
        return None

    translated_citation: Dict[str, Any] = {"sources": translated_sources}
    start_index = _index(citation.get("start"))
    end_index = _index(citation.get("end"))
    if start_index is not None:
        translated_citation["start_index"] = start_index
    if end_index is not None:
        translated_citation["end_index"] = end_index
    content_index = _index(citation.get("content_index"))
    if content_index is not None:
        translated_citation["content_index"] = content_index
    return translated_citation


def _translate_source(source: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(source, dict):
        return None

    source_type = _source_type(source)
    if source_type not in _ALLOWED_SOURCE_TYPES:
        return None

    document = source.get("document") if source_type == _DOCUMENT_SOURCE else None
    if not isinstance(document, dict):
        document = None

    source_id = _source_id(source.get("id"))
    if source_id is None and document is not None:
        source_id = _source_id(document.get("id"))
    if source_id is None:
        return None

    translated: Dict[str, Any] = {
        "type": source_type,
        "id": source_id,
    }
    if source_type != _DOCUMENT_SOURCE:
        return translated

    title = _nonempty_str(source.get("title"))
    url = _http_url(source.get("url"))
    if document is not None:
        if title is None:
            title = _nonempty_str(document.get("title"))
        if url is None:
            url = _http_url(document.get("url"))
    if title is not None:
        translated["title"] = title
    if url is not None:
        translated["url"] = url
    return translated


def _document_url_annotation(
    citation: dict,
    source: Dict[str, Any],
) -> Optional[ChatCompletionAnnotation]:
    """Emit a Responses-valid url_citation only for a real document URL."""
    if source.get("type") != _DOCUMENT_SOURCE:
        return None

    start_index = _index(citation.get("start"))
    end_index = _index(citation.get("end"))
    title = source.get("title")
    url = source.get("url")
    if (
        start_index is None
        or end_index is None
        or end_index < start_index
        or not isinstance(title, str)
        or not isinstance(url, str)
    ):
        return None

    url_citation: ChatCompletionAnnotationURLCitation = {
        "start_index": start_index,
        "end_index": end_index,
        "title": title,
        "url": url,
    }
    annotation: ChatCompletionAnnotation = {
        "type": "url_citation",
        "url_citation": url_citation,
    }
    return annotation


def _source_type(source: dict) -> Optional[str]:
    raw_type = source.get("type")
    if raw_type is None:
        if isinstance(source.get("document"), dict):
            return _DOCUMENT_SOURCE
        if "tool_output" in source:
            return _TOOL_SOURCE
        return None
    if not isinstance(raw_type, str) or raw_type not in _ALLOWED_SOURCE_TYPES:
        return None
    return raw_type


def _index(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _source_id(value: Any) -> Optional[str]:
    """Keep a nonempty source id exactly, including surrounding whitespace."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _nonempty_str(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text


def _http_url(value: Any) -> Optional[str]:
    """Accept an absolute http(s) URL. Reject synthesized and credentialed URLs."""
    url = _nonempty_str(value)
    if url is None or any(char.isspace() for char in url):
        return None
    lowered = url.lower()
    if lowered.startswith("https://"):
        remainder = url[8:]
    elif lowered.startswith("http://"):
        remainder = url[7:]
    else:
        return None
    authority = remainder.split("/", 1)[0]
    if not authority or authority.startswith(".") or "@" in authority:
        return None
    return url
