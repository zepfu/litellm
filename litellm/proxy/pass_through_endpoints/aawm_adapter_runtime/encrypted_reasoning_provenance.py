"""OPENAI-006 encrypted-reasoning item provenance and OpenAI egress compatibility.

Provider-neutral helpers that:
1. Stamp safe producer provenance onto Responses reasoning items that carry
   encrypted_content (item-level; no encrypted bytes in logs/metadata).
2. Guard OpenAI/Codex Responses egress so known foreign-provider or
   incompatible state-format encrypted reasoning fails closed before upstream
   send with a structured fresh-dispatch / non-resumable outcome.
3. Preserve same-compatible-provider encrypted bytes byte-for-byte on egress.
4. Treat OpenAI account lane as non-cryptographic: account1-to-account2 alone
   must not reject stateless encrypted content.

D1-612 remains sole session/provider/model/route/account owner.
OPENAI-007 remains sole function_call id/call_id owner.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from fastapi import HTTPException

# Exact compatibility source used by the guard and tests.
# Compatibility is producer_provider_family + encrypted_state_format only.
# Account lane is audit-only and never part of the cryptographic boundary.
ENCRYPTED_REASONING_COMPATIBILITY_SOURCE = (
    "producer_provider_family+encrypted_state_format"
)

PROVENANCE_ITEM_FIELD = "aawm_encrypted_reasoning_provenance"
_WRAP_PREFIX = "aawm_erp:"
_PROVENANCE_VERSION = 1

# Wire/state families for encrypted reasoning blobs.
STATE_FORMAT_OPENAI_ENCRYPTED_REASONING = "openai_encrypted_reasoning"
STATE_FORMAT_XAI_ENCRYPTED_REASONING = "xai_encrypted_reasoning"
STATE_FORMAT_ANTHROPIC_ENCRYPTED_REASONING = "anthropic_encrypted_reasoning"
STATE_FORMAT_FOREIGN_ENCRYPTED_REASONING = "foreign_encrypted_reasoning"

_OPENAI_PROVIDER_ALIASES = frozenset(
    {
        "openai",
        "chatgpt",
        "codex",
        "azure",
        "azure_ai",
        "azure_openai",
    }
)
_XAI_PROVIDER_ALIASES = frozenset(
    {
        "xai",
        "oa_xai",
        "grok",
        "xai_oauth",
    }
)
_ANTHROPIC_PROVIDER_ALIASES = frozenset(
    {
        "anthropic",
        "claude",
    }
)

# OpenAI Responses egress accepts only OpenAI-family producer + OpenAI
# encrypted-reasoning state format.
_OPENAI_EGRESS_COMPATIBLE = frozenset(
    {
        ("openai", STATE_FORMAT_OPENAI_ENCRYPTED_REASONING),
    }
)

_SAFE_PROVENANCE_KEYS = (
    "version",
    "producer_provider_family",
    "producer_provider",
    "producer_model",
    "producer_route_family",
    "encrypted_state_format",
    "compatibility_source",
    "account_label",
    "account_lane",
    # account_hash is a one-way safe digest already used elsewhere; never raw id.
    "account_hash",
)


def normalize_producer_provider_family(provider: Any) -> str:
    """Map a concrete provider / credential family onto a coarse family."""
    if provider is None:
        return "unknown"
    text = str(provider).strip().lower()
    if not text:
        return "unknown"
    # Strip common prefixes used in model ids (oa_xai/grok-...).
    if "/" in text:
        text = text.split("/", 1)[0]
    if text in _OPENAI_PROVIDER_ALIASES or text.startswith("openai"):
        return "openai"
    if text in _XAI_PROVIDER_ALIASES or text.startswith("xai") or text.startswith(
        "oa_xai"
    ):
        return "xai"
    if text in _ANTHROPIC_PROVIDER_ALIASES or text.startswith("anthropic"):
        return "anthropic"
    if text.startswith("azure"):
        return "openai"
    return text.replace(" ", "_")


def infer_encrypted_state_format(
    *,
    producer_provider_family: str,
    state_format: Any = None,
    route_family: Any = None,
) -> str:
    """Infer encrypted-reasoning state-format family from producer context."""
    family = normalize_producer_provider_family(producer_provider_family)
    explicit = str(state_format or "").strip().lower()
    route = str(route_family or "").strip().lower()

    if "anthropic" in explicit or family == "anthropic":
        return STATE_FORMAT_ANTHROPIC_ENCRYPTED_REASONING
    if (
        family == "xai"
        or "xai" in explicit
        or "xai" in route
        or "grok" in route
        or "oa_xai" in route
    ):
        return STATE_FORMAT_XAI_ENCRYPTED_REASONING
    if family == "openai" or "openai" in explicit or "codex" in route:
        return STATE_FORMAT_OPENAI_ENCRYPTED_REASONING
    if family and family != "unknown":
        return STATE_FORMAT_FOREIGN_ENCRYPTED_REASONING
    return STATE_FORMAT_FOREIGN_ENCRYPTED_REASONING


def build_encrypted_reasoning_provenance(
    *,
    producer_provider: Any = None,
    producer_model: Any = None,
    producer_route_family: Any = None,
    state_format: Any = None,
    account_label: Any = None,
    account_lane: Any = None,
    account_hash: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build safe item-level producer provenance (no secrets / no ciphertext)."""
    provider_family = normalize_producer_provider_family(producer_provider)
    encrypted_state_format = infer_encrypted_state_format(
        producer_provider_family=provider_family,
        state_format=state_format,
        route_family=producer_route_family,
    )
    provenance: dict[str, Any] = {
        "version": _PROVENANCE_VERSION,
        "producer_provider_family": provider_family,
        "encrypted_state_format": encrypted_state_format,
        "compatibility_source": ENCRYPTED_REASONING_COMPATIBILITY_SOURCE,
    }
    if producer_provider is not None and str(producer_provider).strip():
        provenance["producer_provider"] = str(producer_provider).strip()
    if producer_model is not None and str(producer_model).strip():
        provenance["producer_model"] = str(producer_model).strip()
    if producer_route_family is not None and str(producer_route_family).strip():
        provenance["producer_route_family"] = str(producer_route_family).strip()
    # Account fields are audit-only; never part of compatibility.
    if account_label is not None and str(account_label).strip():
        provenance["account_label"] = str(account_label).strip()
    if account_lane is not None and str(account_lane).strip():
        provenance["account_lane"] = str(account_lane).strip()
    if account_hash is not None and str(account_hash).strip():
        # Already a safe digest in D1-612 / codex_oauth inventory paths.
        provenance["account_hash"] = str(account_hash).strip()
    if isinstance(extra, Mapping):
        for key, value in extra.items():
            if key in provenance or key not in _SAFE_PROVENANCE_KEYS:
                continue
            if value is None:
                continue
            text = str(value).strip()
            if text:
                provenance[key] = text
    return provenance


def sanitize_encrypted_reasoning_provenance(
    value: Any,
) -> Optional[dict[str, Any]]:
    """Return a safe provenance dict or None; never includes ciphertext."""
    if not isinstance(value, Mapping):
        return None
    cleaned: dict[str, Any] = {}
    for key in _SAFE_PROVENANCE_KEYS:
        if key not in value:
            continue
        raw = value.get(key)
        if raw is None:
            continue
        if key == "version":
            try:
                cleaned[key] = int(raw)
            except (TypeError, ValueError):
                cleaned[key] = _PROVENANCE_VERSION
            continue
        text = str(raw).strip()
        if text:
            cleaned[key] = text
    if "producer_provider_family" not in cleaned and "producer_provider" in cleaned:
        cleaned["producer_provider_family"] = normalize_producer_provider_family(
            cleaned.get("producer_provider")
        )
    if "encrypted_state_format" not in cleaned:
        cleaned["encrypted_state_format"] = infer_encrypted_state_format(
            producer_provider_family=str(
                cleaned.get("producer_provider_family") or "unknown"
            ),
            route_family=cleaned.get("producer_route_family"),
        )
    if "compatibility_source" not in cleaned:
        cleaned["compatibility_source"] = ENCRYPTED_REASONING_COMPATIBILITY_SOURCE
    if "producer_provider_family" not in cleaned:
        return None
    return cleaned


def _encode_wrap_metadata(provenance: Mapping[str, Any]) -> str:
    payload = {
        key: provenance[key]
        for key in _SAFE_PROVENANCE_KEYS
        if key in provenance and provenance[key] is not None
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def wrap_encrypted_content_with_provenance(
    encrypted_content: str,
    provenance: Mapping[str, Any],
) -> str:
    """Embed safe provenance ahead of ciphertext without altering ciphertext body.

    Format: ``aawm_erp:{base64(json)};{original_encrypted_content}``

    Idempotent: already-wrapped content is not double-wrapped. The original
    encrypted bytes after the first ``;`` separator are preserved exactly.
    """
    if not isinstance(encrypted_content, str) or not encrypted_content:
        return encrypted_content
    if encrypted_content.startswith(_WRAP_PREFIX):
        # Refresh metadata but keep original ciphertext byte-for-byte.
        _, original = unwrap_encrypted_content_with_provenance(encrypted_content)
        safe = sanitize_encrypted_reasoning_provenance(provenance) or dict(provenance)
        return f"{_WRAP_PREFIX}{_encode_wrap_metadata(safe)};{original}"
    safe = sanitize_encrypted_reasoning_provenance(provenance) or dict(provenance)
    return f"{_WRAP_PREFIX}{_encode_wrap_metadata(safe)};{encrypted_content}"


def unwrap_encrypted_content_with_provenance(
    wrapped_content: str,
) -> tuple[Optional[dict[str, Any]], str]:
    """Split wrapped content into (provenance_or_None, original_ciphertext)."""
    if not isinstance(wrapped_content, str) or not wrapped_content.startswith(
        _WRAP_PREFIX
    ):
        return None, wrapped_content
    try:
        rest = wrapped_content[len(_WRAP_PREFIX) :]
        meta_b64, original = rest.split(";", 1)
        missing = len(meta_b64) % 4
        if missing:
            meta_b64 += "=" * (4 - missing)
        decoded = base64.b64decode(meta_b64.encode("ascii")).decode("utf-8")
        parsed = json.loads(decoded)
        return sanitize_encrypted_reasoning_provenance(parsed), original
    except Exception:
        return None, wrapped_content


def _item_has_encrypted_reasoning(item: Any) -> bool:
    if not isinstance(item, MutableMapping) and not isinstance(item, Mapping):
        # object form
        item_type = getattr(item, "type", None)
        encrypted = getattr(item, "encrypted_content", None)
        return item_type == "reasoning" and isinstance(encrypted, str) and bool(encrypted)
    item_type = item.get("type")
    encrypted = item.get("encrypted_content")
    return item_type == "reasoning" and isinstance(encrypted, str) and bool(encrypted)


def extract_item_encrypted_reasoning_provenance(
    item: Any,
) -> Optional[dict[str, Any]]:
    """Read provenance from item sidecar and/or ciphertext wrap metadata."""
    sidecar = None
    encrypted = None
    if isinstance(item, Mapping):
        sidecar = item.get(PROVENANCE_ITEM_FIELD)
        encrypted = item.get("encrypted_content")
    else:
        sidecar = getattr(item, PROVENANCE_ITEM_FIELD, None)
        encrypted = getattr(item, "encrypted_content", None)

    from_sidecar = sanitize_encrypted_reasoning_provenance(sidecar)
    from_wrap = None
    if isinstance(encrypted, str):
        from_wrap, _ = unwrap_encrypted_content_with_provenance(encrypted)
    if from_sidecar and from_wrap:
        # Prefer explicit sidecar then fill gaps from wrap.
        merged = dict(from_wrap)
        merged.update(from_sidecar)
        return sanitize_encrypted_reasoning_provenance(merged)
    return from_sidecar or from_wrap


def stamp_encrypted_reasoning_provenance_on_item(
    item: Any,
    provenance: Mapping[str, Any],
) -> Any:
    """Stamp one reasoning item in-place when possible; return the item."""
    safe = sanitize_encrypted_reasoning_provenance(provenance)
    if safe is None:
        return item
    if isinstance(item, dict):
        if not _item_has_encrypted_reasoning(item):
            return item
        encrypted = item.get("encrypted_content")
        if isinstance(encrypted, str) and encrypted:
            # Preserve original ciphertext inside the wrap.
            item["encrypted_content"] = wrap_encrypted_content_with_provenance(
                encrypted, safe
            )
        item[PROVENANCE_ITEM_FIELD] = dict(safe)
        return item
    # object with attributes
    if not _item_has_encrypted_reasoning(item):
        return item
    encrypted = getattr(item, "encrypted_content", None)
    if isinstance(encrypted, str) and encrypted:
        try:
            setattr(
                item,
                "encrypted_content",
                wrap_encrypted_content_with_provenance(encrypted, safe),
            )
        except Exception:
            pass
    try:
        setattr(item, PROVENANCE_ITEM_FIELD, dict(safe))
    except Exception:
        pass
    return item


def iter_encrypted_reasoning_items(container: Any) -> list[Any]:
    """Collect reasoning items that carry encrypted_content from input/output."""
    items: list[Any] = []
    if isinstance(container, Mapping):
        for key in ("input", "output"):
            value = container.get(key)
            if isinstance(value, list):
                for entry in value:
                    if _item_has_encrypted_reasoning(entry):
                        items.append(entry)
        # Also accept a bare list under no key when callers pass input list.
    elif isinstance(container, list):
        for entry in container:
            if _item_has_encrypted_reasoning(entry):
                items.append(entry)
    return items


def stamp_encrypted_reasoning_provenance_in_response(
    response: Any,
    provenance: Mapping[str, Any],
) -> Any:
    """Stamp all encrypted reasoning items in a Responses payload/output."""
    safe = sanitize_encrypted_reasoning_provenance(provenance)
    if safe is None:
        return response
    if isinstance(response, Mapping):
        output = response.get("output")
        if isinstance(output, list):
            for index, item in enumerate(output):
                if isinstance(item, dict) and _item_has_encrypted_reasoning(item):
                    output[index] = stamp_encrypted_reasoning_provenance_on_item(
                        item, safe
                    )
                elif _item_has_encrypted_reasoning(item):
                    stamp_encrypted_reasoning_provenance_on_item(item, safe)
        return response
    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            if _item_has_encrypted_reasoning(item):
                stamp_encrypted_reasoning_provenance_on_item(item, safe)
    return response


def prepare_encrypted_reasoning_items_for_openai_egress(
    request_body: Mapping[str, Any] | dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Unwrap stamped ciphertext for egress while collecting disposition metadata.

    Returns ``(updated_body, disposition_metadata)``. Same-compatible-provider
    ciphertext is restored byte-for-byte. Does not log or return ciphertext.
    Does not raise; callers run the compatibility guard separately.
    """
    if not isinstance(request_body, dict):
        return dict(request_body) if isinstance(request_body, Mapping) else {}, {
            "encrypted_reasoning_item_count": 0,
            "encrypted_reasoning_disposition": "absent",
            "encrypted_reasoning_compatibility_source": (
                ENCRYPTED_REASONING_COMPATIBILITY_SOURCE
            ),
        }

    input_items = request_body.get("input")
    if not isinstance(input_items, list):
        return request_body, {
            "encrypted_reasoning_item_count": 0,
            "encrypted_reasoning_disposition": "absent",
            "encrypted_reasoning_compatibility_source": (
                ENCRYPTED_REASONING_COMPATIBILITY_SOURCE
            ),
        }

    changed = False
    producers: list[str] = []
    state_formats: list[str] = []
    item_summaries: list[dict[str, Any]] = []
    normalized_input: list[Any] = []

    for item in input_items:
        if not isinstance(item, dict) or not _item_has_encrypted_reasoning(item):
            normalized_input.append(item)
            continue

        provenance = extract_item_encrypted_reasoning_provenance(item)
        encrypted = item.get("encrypted_content")
        original = encrypted
        wrap_prov = None
        if isinstance(encrypted, str):
            wrap_prov, original = unwrap_encrypted_content_with_provenance(encrypted)

        if provenance is None and wrap_prov is not None:
            provenance = wrap_prov

        clean_item = dict(item)
        if isinstance(encrypted, str) and original != encrypted:
            clean_item["encrypted_content"] = original
            changed = True
        # Keep sidecar for local observability on the request object only when
        # already present; do not invent ciphertext-bearing fields.
        if provenance is not None and PROVENANCE_ITEM_FIELD not in clean_item:
            clean_item[PROVENANCE_ITEM_FIELD] = dict(provenance)
            changed = True

        # OpenAI rejects unknown fields on reasoning items; strip sidecar before
        # upstream send while retaining disposition from the extracted copy.
        if PROVENANCE_ITEM_FIELD in clean_item:
            clean_item.pop(PROVENANCE_ITEM_FIELD, None)
            changed = True

        normalized_input.append(clean_item)

        family = (
            str(provenance.get("producer_provider_family"))
            if isinstance(provenance, Mapping)
            else "unknown"
        )
        state_fmt = (
            str(provenance.get("encrypted_state_format"))
            if isinstance(provenance, Mapping)
            else "unknown"
        )
        producers.append(family)
        state_formats.append(state_fmt)
        summary: dict[str, Any] = {
            "producer_provider_family": family,
            "encrypted_state_format": state_fmt,
            "compatibility_source": ENCRYPTED_REASONING_COMPATIBILITY_SOURCE,
        }
        if isinstance(provenance, Mapping):
            for key in (
                "producer_provider",
                "producer_model",
                "producer_route_family",
                "account_label",
                "account_lane",
            ):
                value = provenance.get(key)
                if value is not None and str(value).strip():
                    summary[key] = str(value).strip()
        item_id = clean_item.get("id")
        if isinstance(item_id, str) and item_id and not item_id.startswith(
            ("gAAA", "aawm_")
        ):
            # Item ids (rs_*) are safe correlation tokens; never ciphertext.
            summary["item_id"] = item_id
        item_summaries.append(summary)

    disposition = {
        "encrypted_reasoning_item_count": len(item_summaries),
        "encrypted_reasoning_disposition": (
            "present" if item_summaries else "absent"
        ),
        "encrypted_reasoning_compatibility_source": (
            ENCRYPTED_REASONING_COMPATIBILITY_SOURCE
        ),
        "encrypted_reasoning_producer_provider_families": sorted(set(producers)),
        "encrypted_reasoning_state_formats": sorted(set(state_formats)),
        "encrypted_reasoning_items": item_summaries,
    }

    if not changed:
        return request_body, disposition
    updated = dict(request_body)
    updated["input"] = normalized_input
    return updated, disposition


def is_openai_encrypted_reasoning_compatible(
    provenance: Optional[Mapping[str, Any]],
) -> bool:
    """Return True when provenance may be sent to OpenAI Responses egress."""
    if not isinstance(provenance, Mapping):
        # Legacy unstamped items: treat as OpenAI-compatible so same-provider
        # continuations that predate stamping are not broken. Foreign producers
        # are required to stamp on the way out.
        return True
    raw_family = provenance.get("producer_provider_family") or provenance.get(
        "producer_provider"
    )
    raw_state = provenance.get("encrypted_state_format")
    # Summaries for unstamped legacy items use unknown/unknown; allow them so
    # pre-OPENAI-006 OpenAI continuations keep working. Known foreign families
    # still fail closed.
    if (
        (raw_family is None or str(raw_family).strip().lower() in {"", "unknown"})
        and (
            raw_state is None
            or str(raw_state).strip().lower() in {"", "unknown"}
        )
    ):
        return True
    family = normalize_producer_provider_family(raw_family)
    state_fmt = str(
        raw_state or infer_encrypted_state_format(producer_provider_family=family)
    ).strip()
    return (family, state_fmt) in _OPENAI_EGRESS_COMPATIBLE


def build_encrypted_reasoning_disposition_metadata(
    *,
    disposition: str,
    items: Sequence[Mapping[str, Any]] | None = None,
    mismatch_reason: Optional[str] = None,
    compatibility_ok: Optional[bool] = None,
) -> dict[str, Any]:
    """Safe request/session/attempt metadata — never ciphertext or secrets."""
    item_list = [dict(item) for item in (items or []) if isinstance(item, Mapping)]
    producers = sorted(
        {
            str(item.get("producer_provider_family"))
            for item in item_list
            if item.get("producer_provider_family")
        }
    )
    state_formats = sorted(
        {
            str(item.get("encrypted_state_format"))
            for item in item_list
            if item.get("encrypted_state_format")
        }
    )
    meta: dict[str, Any] = {
        "encrypted_reasoning_disposition": disposition,
        "encrypted_reasoning_item_count": len(item_list),
        "encrypted_reasoning_compatibility_source": (
            ENCRYPTED_REASONING_COMPATIBILITY_SOURCE
        ),
        "encrypted_reasoning_producer_provider_families": producers,
        "encrypted_reasoning_state_formats": state_formats,
        "encrypted_reasoning_items": item_list,
    }
    if mismatch_reason:
        meta["encrypted_reasoning_mismatch_reason"] = mismatch_reason
    if compatibility_ok is not None:
        meta["encrypted_reasoning_compatible"] = bool(compatibility_ok)
    return meta


def raise_openai_encrypted_reasoning_redispatch_required(
    *,
    mismatch_reason: str,
    items: Sequence[Mapping[str, Any]] | None = None,
    session_identity: Any = None,
    target_provider: str = "openai",
    failure_phase: str = "encrypted_reasoning_openai_pre_egress",
) -> None:
    """Fail before OpenAI egress with structured fresh-dispatch outcome."""
    item_list = [dict(item) for item in (items or []) if isinstance(item, Mapping)]
    disposition = build_encrypted_reasoning_disposition_metadata(
        disposition="rejected_incompatible_producer",
        items=item_list,
        mismatch_reason=mismatch_reason,
        compatibility_ok=False,
    )
    detail: dict[str, Any] = {
        "error": {
            "message": (
                "Encrypted reasoning state is not compatible with the target "
                "OpenAI/Codex route. Fresh dispatch is required; do not resume "
                "this session against an incompatible producer/state-format."
            ),
            "type": "invalid_request_error",
            "code": "aawm_encrypted_reasoning_redispatch_required",
        },
        "redispatch_required": True,
        "redispatch_reason": mismatch_reason,
        "failure_phase": failure_phase,
        "attempted_provider_call": False,
        "non_resumable": True,
        "fresh_dispatch_required": True,
        "target_provider": target_provider,
        "encrypted_reasoning": disposition,
        "compatibility_source": ENCRYPTED_REASONING_COMPATIBILITY_SOURCE,
    }
    if session_identity is not None and str(session_identity).strip():
        detail["canonical_session_identity"] = str(session_identity).strip()
    raise HTTPException(status_code=409, detail=detail)


def guard_openai_encrypted_reasoning_egress(
    request_body: Mapping[str, Any] | dict[str, Any],
    *,
    session_identity: Any = None,
    target_provider: str = "openai",
    target_route_family: Any = None,
    failure_phase: str = "encrypted_reasoning_openai_pre_egress",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare body and fail closed on known-incompatible encrypted reasoning.

    Returns ``(prepared_body, disposition_metadata)`` when compatible.
    """
    prepared, disposition = prepare_encrypted_reasoning_items_for_openai_egress(
        request_body if isinstance(request_body, dict) else dict(request_body or {})
    )
    items_meta = disposition.get("encrypted_reasoning_items") or []
    if not items_meta:
        disposition = build_encrypted_reasoning_disposition_metadata(
            disposition="absent",
            items=[],
            compatibility_ok=True,
        )
        return prepared, disposition

    incompatible: list[dict[str, Any]] = []
    for item_meta in items_meta:
        if not isinstance(item_meta, Mapping):
            continue
        if not is_openai_encrypted_reasoning_compatible(item_meta):
            incompatible.append(dict(item_meta))

    if incompatible:
        families = sorted(
            {
                str(item.get("producer_provider_family") or "unknown")
                for item in incompatible
            }
        )
        state_formats = sorted(
            {
                str(item.get("encrypted_state_format") or "unknown")
                for item in incompatible
            }
        )
        reason = (
            "incompatible_encrypted_reasoning_producer:"
            + ",".join(families)
            + ";state_format:"
            + ",".join(state_formats)
        )
        raise_openai_encrypted_reasoning_redispatch_required(
            mismatch_reason=reason,
            items=incompatible,
            session_identity=session_identity,
            target_provider=target_provider,
            failure_phase=failure_phase,
        )

    disposition = build_encrypted_reasoning_disposition_metadata(
        disposition="allowed_compatible",
        items=[dict(item) for item in items_meta if isinstance(item, Mapping)],
        compatibility_ok=True,
    )
    # target route is audit-only
    if target_route_family is not None and str(target_route_family).strip():
        disposition["encrypted_reasoning_target_route_family"] = str(
            target_route_family
        ).strip()
    return prepared, disposition


def merge_encrypted_reasoning_disposition_into_request_body(
    request_body: dict[str, Any],
    disposition: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach safe disposition fields under litellm_metadata (no ciphertext)."""
    if not isinstance(request_body, dict):
        return request_body
    updated = dict(request_body)
    metadata = dict(updated.get("litellm_metadata") or {})
    for key, value in disposition.items():
        if key == "encrypted_reasoning_items":
            # Keep compact item summaries; already safe.
            metadata[key] = value
            continue
        metadata[key] = value
    tags = metadata.get("tags")
    if not isinstance(tags, list):
        tags = []
    else:
        tags = list(tags)
    disposition_name = str(disposition.get("encrypted_reasoning_disposition") or "")
    tag = f"encrypted-reasoning:{disposition_name or 'unknown'}"
    if tag not in tags:
        tags.append(tag)
    for family in disposition.get("encrypted_reasoning_producer_provider_families") or []:
        family_tag = f"encrypted-reasoning-producer:{family}"
        if family_tag not in tags:
            tags.append(family_tag)
    metadata["tags"] = tags
    updated["litellm_metadata"] = metadata
    return updated


def build_producer_provenance_from_egress_context(
    *,
    custom_llm_provider: Any = None,
    egress_credential_family: Any = None,
    expected_target_family: Any = None,
    request_body: Optional[Mapping[str, Any]] = None,
    route_family: Any = None,
    account_label: Any = None,
    account_lane: Any = None,
    account_hash: Any = None,
) -> dict[str, Any]:
    """Construct producer provenance for stamping outbound encrypted items."""
    body = request_body if isinstance(request_body, Mapping) else {}
    model = body.get("model") if isinstance(body, Mapping) else None
    provider = (
        custom_llm_provider
        or expected_target_family
        or egress_credential_family
        or "openai"
    )
    resolved_route = (
        route_family
        or egress_credential_family
        or expected_target_family
        or provider
    )
    # Prefer bound inventory identity from metadata when present.
    metadata = body.get("litellm_metadata") if isinstance(body, Mapping) else None
    if isinstance(metadata, Mapping):
        account_label = account_label or metadata.get(
            "codex_auto_agent_selected_account_label"
        ) or metadata.get("account_label")
        account_lane = account_lane or metadata.get(
            "codex_auto_agent_selected_account_lane"
        ) or metadata.get("account_lane") or metadata.get("codex_oauth_lane_key")
        account_hash = account_hash or metadata.get(
            "codex_auto_agent_selected_account_hash"
        ) or metadata.get("account_hash")
        if model is None:
            model = metadata.get("codex_auto_agent_selected_model") or model
        if not route_family:
            resolved_route = (
                metadata.get("codex_auto_agent_selected_route_family")
                or metadata.get("passthrough_route_family")
                or resolved_route
            )
        provider = (
            metadata.get("codex_auto_agent_selected_provider")
            or provider
        )
    return build_encrypted_reasoning_provenance(
        producer_provider=provider,
        producer_model=model,
        producer_route_family=resolved_route,
        account_label=account_label,
        account_lane=account_lane,
        account_hash=account_hash,
    )


def is_openai_responses_egress(
    *,
    custom_llm_provider: Any = None,
    egress_credential_family: Any = None,
    expected_target_family: Any = None,
    url_path: Any = None,
) -> bool:
    """True when the outbound call is OpenAI/Codex Responses shaped."""
    path = str(url_path or "").lower()
    if "responses" not in path:
        return False
    provider = normalize_producer_provider_family(
        custom_llm_provider or expected_target_family or egress_credential_family
    )
    # Guard only true OpenAI family egress. xAI/Grok responses paths produce
    # foreign encrypted state and must not apply the OpenAI rejection rule.
    if provider != "openai":
        # expected_target_family openai with credential codex_oauth still openai
        target = normalize_producer_provider_family(expected_target_family)
        cred = normalize_producer_provider_family(egress_credential_family)
        if target == "openai" or cred in {"openai", "codex"}:
            return True
        if str(egress_credential_family or "").lower() in {
            "codex_oauth",
            "openai",
            "chatgpt",
        }:
            return True
        if str(expected_target_family or "").lower() in {
            "openai",
            "codex_oauth",
            "chatgpt",
        }:
            return True
        return False
    return True



def stamp_encrypted_reasoning_in_responses_sse_chunk(
    chunk: bytes,
    *,
    request_body: Optional[Mapping[str, Any]] = None,
    custom_llm_provider: Any = None,
    egress_credential_family: Any = None,
    expected_target_family: Any = None,
) -> bytes:
    """Stamp producer provenance onto encrypted reasoning in SSE data lines.

    Best-effort: returns the original chunk unchanged on parse failure or when
    no encrypted reasoning item is present. Never logs ciphertext.
    """
    if not chunk:
        return chunk
    try:
        decoded = chunk.decode("utf-8")
    except Exception:
        return chunk
    if "encrypted_content" not in decoded or "reasoning" not in decoded:
        return chunk

    provenance = build_producer_provenance_from_egress_context(
        custom_llm_provider=custom_llm_provider,
        egress_credential_family=egress_credential_family,
        expected_target_family=expected_target_family,
        request_body=request_body if isinstance(request_body, Mapping) else None,
    )

    any_changed = False
    out_lines: list[str] = []
    for line in decoded.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped.startswith("data:"):
            out_lines.append(line)
            continue
        payload = stripped[5:].strip()
        if not payload or payload == "[DONE]":
            out_lines.append(line)
            continue
        try:
            event = json.loads(payload)
        except Exception:
            out_lines.append(line)
            continue
        if not isinstance(event, dict):
            out_lines.append(line)
            continue

        line_changed = _stamp_encrypted_reasoning_in_sse_event(event, provenance)
        if not line_changed:
            out_lines.append(line)
            continue
        any_changed = True
        if line.endswith('\r\n'):
            ending = '\r\n'
        elif line.endswith('\n'):
            ending = '\n'
        else:
            ending = ""
        out_lines.append(
            "data: "
            + json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            + ending
        )

    if not any_changed:
        return chunk
    rebuilt = "".join(out_lines)
    if decoded.endswith('\n') and not rebuilt.endswith('\n'):
        rebuilt += '\n'
    return rebuilt.encode("utf-8")


def _stamp_encrypted_reasoning_in_sse_event(
    event: dict[str, Any],
    provenance: Mapping[str, Any],
) -> bool:
    """Mutate one SSE event dict when it carries encrypted reasoning. Return changed."""
    changed = False
    event_type = event.get("type")
    if event_type in {"response.output_item.added", "response.output_item.done"}:
        item = event.get("item")
        if isinstance(item, dict) and _item_has_encrypted_reasoning(item):
            before = item.get("encrypted_content")
            stamp_encrypted_reasoning_provenance_on_item(item, provenance)
            event["item"] = item
            if item.get("encrypted_content") != before or item.get(PROVENANCE_ITEM_FIELD):
                changed = True
        return changed
    if event_type == "response.completed":
        response_obj = event.get("response")
        output = response_obj.get("output") if isinstance(response_obj, dict) else None
        if not isinstance(output, list):
            return False
        for index, item in enumerate(output):
            if isinstance(item, dict) and _item_has_encrypted_reasoning(item):
                before = item.get("encrypted_content")
                stamp_encrypted_reasoning_provenance_on_item(item, provenance)
                output[index] = item
                if item.get("encrypted_content") != before or item.get(
                    PROVENANCE_ITEM_FIELD
                ):
                    changed = True
    return changed



__all__ = [
    "ENCRYPTED_REASONING_COMPATIBILITY_SOURCE",
    "PROVENANCE_ITEM_FIELD",
    "STATE_FORMAT_OPENAI_ENCRYPTED_REASONING",
    "STATE_FORMAT_XAI_ENCRYPTED_REASONING",
    "STATE_FORMAT_ANTHROPIC_ENCRYPTED_REASONING",
    "STATE_FORMAT_FOREIGN_ENCRYPTED_REASONING",
    "normalize_producer_provider_family",
    "infer_encrypted_state_format",
    "build_encrypted_reasoning_provenance",
    "sanitize_encrypted_reasoning_provenance",
    "wrap_encrypted_content_with_provenance",
    "unwrap_encrypted_content_with_provenance",
    "extract_item_encrypted_reasoning_provenance",
    "stamp_encrypted_reasoning_provenance_on_item",
    "stamp_encrypted_reasoning_provenance_in_response",
    "iter_encrypted_reasoning_items",
    "prepare_encrypted_reasoning_items_for_openai_egress",
    "is_openai_encrypted_reasoning_compatible",
    "build_encrypted_reasoning_disposition_metadata",
    "raise_openai_encrypted_reasoning_redispatch_required",
    "guard_openai_encrypted_reasoning_egress",
    "merge_encrypted_reasoning_disposition_into_request_body",
    "build_producer_provenance_from_egress_context",
    "is_openai_responses_egress",
    "stamp_encrypted_reasoning_in_responses_sse_chunk",
]
