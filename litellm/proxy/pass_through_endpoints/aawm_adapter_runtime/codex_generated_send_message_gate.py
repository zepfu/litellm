"""Request-local gate for generated Codex V2 send_message calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from litellm.responses.function_name_sanitization import ResponsesFunctionIdentity

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_collaboration_dispatch import (
    CodexCollaborationDispatchError,
    build_codex_collaboration_wire_aliases,
    canonicalize_generated_codex_send_message_call_arguments,
    collect_codex_collaboration_advertised_tool_names,
    get_bound_codex_collaboration_tool_identities,
    is_codex_collaboration_send_message_identity,
    raise_codex_send_message_output_rejected,
)


CODEX_SEND_MESSAGE_OUTPUT_GATE_ATTR = "_aawm_cfg072_send_message_output_gate"


@dataclass(frozen=True)
class _IdentityResolver:
    aliases: Mapping[str, ResponsesFunctionIdentity]
    pairs: Mapping[tuple[str, Optional[str]], ResponsesFunctionIdentity]

    def resolve(self, name: Any, namespace: Any) -> Optional[ResponsesFunctionIdentity]:
        if not isinstance(name, str) or not name:
            return None
        alias_identity = self.aliases.get(name)
        if alias_identity is not None:
            if namespace is None or namespace in {
                alias_identity.namespace,
                alias_identity.rendered_namespace,
            }:
                return alias_identity
            return None
        if namespace is not None and not isinstance(namespace, str):
            return None
        return self.pairs.get((name, namespace))


@dataclass
class _CallRecord:
    ordinal: int
    identity: ResponsesFunctionIdentity
    original_parts: list[tuple[int, tuple[Any, ...], str]] = field(default_factory=list)
    has_accumulated_original: bool = False
    original_complete: Optional[str] = None
    selected: Optional[str] = None
    complete: bool = False
    complete_surfaces: list[tuple[int, tuple[Any, ...]]] = field(
        default_factory=list
    )


@dataclass
class _QueuedEvent:
    context: Any
    original_payload: Optional[dict[str, Any]]
    payload: Optional[dict[str, Any]]


class CodexSendMessageOutputGate:
    """Select complete generated calls and preserve their ordered SSE frames."""

    def __init__(self, resolver: _IdentityResolver) -> None:
        self._resolver = resolver
        self._records: dict[int, _CallRecord] = {}
        self._id_records: dict[tuple[str, str], int] = {}
        self._pending: set[int] = set()
        self._queued: list[_QueuedEvent] = []
        self._next_ordinal = 0

    @staticmethod
    def _reject(reason: str) -> None:
        raise_codex_send_message_output_rejected(reason)

    @staticmethod
    def _event_ids(
        item: Any,
        event: Any,
    ) -> tuple[tuple[str, str], ...]:
        identifiers: list[tuple[str, str]] = []
        if isinstance(item, dict):
            for field, kind in (("id", "item"), ("call_id", "call")):
                value = item.get(field)
                if isinstance(value, str) and value:
                    identifiers.append((kind, value))
        if isinstance(event, dict):
            for field, kind in (("item_id", "item"), ("call_id", "call")):
                value = event.get(field)
                if isinstance(value, str) and value:
                    identifiers.append((kind, value))
        return tuple(dict.fromkeys(identifiers))

    @staticmethod
    def _set_path(
        value: Any,
        path: tuple[Any, ...],
        replacement: Any,
    ) -> Any:
        if not path:
            return replacement
        head, *tail = path
        if isinstance(value, dict) and isinstance(head, str) and head in value:
            updated = dict(value)
            updated[head] = CodexSendMessageOutputGate._set_path(
                value[head],
                tuple(tail),
                replacement,
            )
            return updated
        if isinstance(value, list) and isinstance(head, int) and 0 <= head < len(value):
            updated = list(value)
            updated[head] = CodexSendMessageOutputGate._set_path(
                value[head],
                tuple(tail),
                replacement,
            )
            return updated
        return value

    def _record_for_ids(
        self,
        identifiers: tuple[tuple[str, str], ...],
    ) -> Optional[_CallRecord]:
        matches = {
            self._id_records[key]
            for key in identifiers
            if key in self._id_records
        }
        if len(matches) > 1:
            self._reject("invalid_envelope")
        return self._records[next(iter(matches))] if matches else None

    def _get_or_create_record(
        self,
        identifiers: tuple[tuple[str, str], ...],
        identity: Optional[ResponsesFunctionIdentity],
    ) -> Optional[_CallRecord]:
        record = self._record_for_ids(identifiers)
        if record is not None:
            if identity is not None and identity != record.identity:
                self._reject("invalid_envelope")
            for key in identifiers:
                owner = self._id_records.get(key)
                if owner is not None and owner != record.ordinal:
                    self._reject("invalid_envelope")
                self._id_records[key] = record.ordinal
            return record
        if identity is None:
            return None
        if not identifiers:
            self._reject("invalid_envelope")
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        record = _CallRecord(ordinal=ordinal, identity=identity)
        self._records[ordinal] = record
        self._pending.add(ordinal)
        for key in identifiers:
            self._id_records[key] = ordinal
        return record

    def _event_identity(
        self,
        event: dict[str, Any],
        item: Any = None,
    ) -> Optional[ResponsesFunctionIdentity]:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            return self._resolver.resolve(
                item.get("name"),
                item.get("namespace"),
            )
        return self._resolver.resolve(
            event.get("name"),
            event.get("namespace"),
        )

    def _check_known_identity(
        self,
        record: _CallRecord,
        event: dict[str, Any],
        item: Any = None,
    ) -> None:
        source = item if isinstance(item, dict) else event
        name = source.get("name")
        namespace = source.get("namespace")
        if isinstance(name, str) and name and self._event_identity(event, item) is None:
            self._reject("invalid_envelope")
        resolved = self._event_identity(event, item)
        if resolved is not None and resolved != record.identity:
            self._reject("invalid_envelope")
        if namespace is not None and not isinstance(namespace, str):
            self._reject("invalid_envelope")

    def _add_original_part(
        self,
        record: _CallRecord,
        queue_index: int,
        path: tuple[Any, ...],
        value: Any,
    ) -> None:
        if not isinstance(value, str):
            self._reject("invalid_envelope")
        if record.complete:
            self._reject("invalid_envelope")
        record.original_parts.append((queue_index, path, value))
        if value:
            record.has_accumulated_original = True

    def _complete_from_snapshots(
        self,
        record: _CallRecord,
        snapshots: list[tuple[tuple[Any, ...], Any]],
        queue_index: int,
    ) -> None:
        if not snapshots:
            if not record.has_accumulated_original:
                self._reject("invalid_envelope")
            self._complete_original(
                record,
                "".join(part[2] for part in record.original_parts),
            )
            return
        for path, value in snapshots:
            if not isinstance(value, str):
                self._reject("invalid_envelope")
            self._complete_original(record, value)
            record.complete_surfaces.append((queue_index, path))

    def _complete_original(self, record: _CallRecord, original: str) -> None:
        if record.original_complete is not None:
            if record.original_complete != original:
                self._reject("invalid_envelope")
            return
        if record.has_accumulated_original:
            accumulated = "".join(part[2] for part in record.original_parts)
            if accumulated != original:
                self._reject("invalid_envelope")
        try:
            selected = canonicalize_generated_codex_send_message_call_arguments(
                original
            )
        except CodexCollaborationDispatchError as exc:
            self._reject(exc.reason)
        record.original_complete = original
        record.selected = selected
        record.complete = True
        self._pending.discard(record.ordinal)

    def _consume_added(
        self,
        payload: dict[str, Any],
        queue_index: int,
    ) -> None:
        item = payload.get("item")
        if not isinstance(item, dict) or item.get("type") != "function_call":
            return
        identifiers = self._event_ids(item, payload)
        identity = self._event_identity(payload, item)
        record = self._record_for_ids(identifiers)
        if record is None and identity is None:
            return
        record = self._get_or_create_record(identifiers, identity)
        if record is None:
            return
        self._check_known_identity(record, payload, item)
        if record.complete:
            self._reject("invalid_envelope")
        if "arguments" in item:
            self._add_original_part(
                record,
                queue_index,
                ("item", "arguments"),
                item.get("arguments"),
            )

    def _consume_delta(
        self,
        payload: dict[str, Any],
        queue_index: int,
    ) -> None:
        identifiers = self._event_ids(None, payload)
        identity = self._event_identity(payload)
        record = self._record_for_ids(identifiers)
        if record is None and identity is None:
            return
        record = self._get_or_create_record(identifiers, identity)
        if record is None:
            return
        self._check_known_identity(record, payload)
        self._add_original_part(
            record,
            queue_index,
            ("delta",),
            payload.get("delta"),
        )

    def _consume_arguments_done(
        self,
        payload: dict[str, Any],
        queue_index: int,
    ) -> None:
        item = payload.get("item")
        identifiers = self._event_ids(item, payload)
        identity = self._event_identity(payload, item)
        record = self._record_for_ids(identifiers)
        if record is None and identity is None:
            return
        record = self._get_or_create_record(identifiers, identity)
        if record is None:
            return
        self._check_known_identity(record, payload, item)
        snapshots: list[tuple[tuple[Any, ...], Any]] = []
        if "arguments" in payload:
            snapshots.append((("arguments",), payload.get("arguments")))
        if isinstance(item, dict) and "arguments" in item:
            snapshots.append((("item", "arguments"), item.get("arguments")))
        self._complete_from_snapshots(record, snapshots, queue_index)

    def _consume_output_item_done(
        self,
        payload: dict[str, Any],
        queue_index: int,
    ) -> None:
        item = payload.get("item")
        if not isinstance(item, dict) or item.get("type") != "function_call":
            return
        identifiers = self._event_ids(item, payload)
        identity = self._event_identity(payload, item)
        record = self._record_for_ids(identifiers)
        if record is None and identity is None:
            return
        record = self._get_or_create_record(identifiers, identity)
        if record is None:
            return
        self._check_known_identity(record, payload, item)
        snapshots = (
            [(("item", "arguments"), item.get("arguments"))]
            if "arguments" in item
            else []
        )
        self._complete_from_snapshots(record, snapshots, queue_index)

    def _consume_terminal_output(
        self,
        payload: dict[str, Any],
        queue_index: int,
    ) -> None:
        response = payload.get("response")
        output = response.get("output") if isinstance(response, dict) else None
        if not isinstance(output, list):
            if self._pending:
                self._reject("invalid_envelope")
            return
        for output_index, item in enumerate(output):
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            identifiers = self._event_ids(item, payload)
            identity = self._event_identity(payload, item)
            record = self._record_for_ids(identifiers)
            if record is None and identity is None:
                continue
            record = self._get_or_create_record(identifiers, identity)
            if record is None:
                continue
            self._check_known_identity(record, payload, item)
            snapshots = (
                [
                    (
                        ("response", "output", output_index, "arguments"),
                        item.get("arguments"),
                    )
                ]
                if "arguments" in item
                else []
            )
            self._complete_from_snapshots(record, snapshots, queue_index)
        if self._pending:
            self._reject("invalid_envelope")

    def _consume_unsuccessful_terminal(
        self,
        payload: dict[str, Any],
        queue_index: int,
    ) -> None:
        for ordinal in tuple(self._pending):
            record = self._records[ordinal]
            for event_index, path, _ in record.original_parts:
                queued = self._queued[event_index]
                if queued.payload is not None:
                    queued.payload = self._set_path(queued.payload, path, "")
            self._pending.discard(ordinal)
        response = payload.get("response")
        output = response.get("output") if isinstance(response, dict) else None
        if not isinstance(output, list):
            return
        for output_index, item in enumerate(output):
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            identifiers = self._event_ids(item, payload)
            identity = self._event_identity(payload, item)
            record = self._record_for_ids(identifiers)
            path = ("response", "output", output_index, "arguments")
            if record is not None and record.selected is not None:
                self._check_known_identity(record, payload, item)
                self._complete_from_snapshots(
                    record,
                    [(path, item.get("arguments"))],
                    queue_index,
                )
                current = self._queued[queue_index].payload
                if current is not None:
                    self._queued[queue_index].payload = self._set_path(
                        current,
                        path,
                        record.selected,
                    )
            elif identity is not None and "arguments" in item:
                current = self._queued[queue_index].payload
                if current is not None:
                    self._queued[queue_index].payload = self._set_path(
                        current,
                        path,
                        "",
                    )

    def _consume(
        self,
        payload: dict[str, Any],
        queue_index: int,
    ) -> None:
        event_type = payload.get("type")
        if event_type == "response.output_item.added":
            self._consume_added(payload, queue_index)
        elif event_type == "response.function_call_arguments.delta":
            self._consume_delta(payload, queue_index)
        elif event_type == "response.function_call_arguments.done":
            self._consume_arguments_done(payload, queue_index)
        elif event_type == "response.output_item.done":
            self._consume_output_item_done(payload, queue_index)
        elif event_type in {"response.completed", "response.done"}:
            response = payload.get("response")
            if isinstance(response, dict) and response.get("status") == "completed":
                self._consume_terminal_output(payload, queue_index)
            else:
                self._consume_unsuccessful_terminal(payload, queue_index)
        elif event_type in {"response.failed", "response.incomplete"}:
            self._consume_unsuccessful_terminal(payload, queue_index)

    def _select_buffered_arguments(self) -> None:
        for record in self._records.values():
            if not record.complete or record.selected is None:
                continue
            if record.original_parts:
                lengths = [len(part[2]) for part in record.original_parts]
                total = sum(lengths)
                selected_length = len(record.selected)
                if total:
                    cumulative = 0
                    start = 0
                    for part_index, (event_index, path, _) in enumerate(
                        record.original_parts
                    ):
                        cumulative += lengths[part_index]
                        end = (
                            selected_length
                            if part_index == len(record.original_parts) - 1
                            else round(selected_length * cumulative / total)
                        )
                        queued = self._queued[event_index]
                        if queued.payload is not None:
                            queued.payload = self._set_path(
                                queued.payload,
                                path,
                                record.selected[start:end],
                            )
                        start = end
                else:
                    event_index, path, _ = record.original_parts[-1]
                    queued = self._queued[event_index]
                    if queued.payload is not None:
                        queued.payload = self._set_path(
                            queued.payload,
                            path,
                            record.selected,
                        )
            for event_index, path in record.complete_surfaces:
                queued = self._queued[event_index]
                if queued.payload is not None:
                    queued.payload = self._set_path(
                        queued.payload,
                        path,
                        record.selected,
                    )

    def process_event(
        self,
        payload: Any,
        context: Any,
    ) -> list[tuple[Any, Optional[dict[str, Any]]]]:
        original = payload if isinstance(payload, dict) else None
        self._queued.append(
            _QueuedEvent(
                context=context,
                original_payload=original,
                payload=original,
            )
        )
        queue_index = len(self._queued) - 1
        if original is not None:
            self._consume(original, queue_index)
        if self._pending:
            return []
        self._select_buffered_arguments()
        completed = [
            (event.context, event.original_payload, event.payload)
            for event in self._queued
        ]
        self._queued = []
        for record in self._records.values():
            record.original_parts.clear()
            record.complete_surfaces.clear()
            record.has_accumulated_original = False
        return [
            (context, payload)
            for context, _original, payload in completed
        ]

    def finish(self) -> None:
        """Reject a stream that ended before a targeted call was complete."""
        if self._pending:
            self._reject("invalid_envelope")

    def gate_output(
        self,
        output: Any,
        *,
        allow_selected_copies: bool = False,
    ) -> Any:
        if not isinstance(output, list):
            return output
        gated: list[Any] = []
        changed = False
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "function_call":
                gated.append(item)
                continue
            identifiers = self._event_ids(item, None)
            identity = self._resolver.resolve(
                item.get("name"),
                item.get("namespace"),
            )
            record = self._record_for_ids(identifiers)
            if record is not None:
                self._check_known_identity(record, item)
                if (
                    allow_selected_copies
                    and record.selected is not None
                    and item.get("arguments") == record.selected
                ):
                    gated.append(item)
                    continue
                snapshots = (
                    [(("arguments",), item.get("arguments"))]
                    if "arguments" in item
                    else []
                )
                self._complete_from_snapshots(record, snapshots, -1)
            elif identity is not None:
                record = self._get_or_create_record(identifiers, identity)
                if record is None:
                    continue
                snapshots = (
                    [(("arguments",), item.get("arguments"))]
                    if "arguments" in item
                    else []
                )
                self._complete_from_snapshots(record, snapshots, -1)
            else:
                gated.append(item)
                continue
            selected = record.selected
            if selected is None:
                self._reject("invalid_envelope")
            if item.get("arguments") == selected:
                gated.append(item)
            else:
                updated = dict(item)
                updated["arguments"] = selected
                gated.append(updated)
                changed = True
        return gated if changed else output


def build_codex_send_message_output_gate(
    request: Any,
    request_body: Any,
    *,
    upstream_to_original_identities: Optional[
        Mapping[str, ResponsesFunctionIdentity]
    ] = None,
    adapted_namespace_by_name: Optional[Mapping[str, str]] = None,
    extra_upstream_to_original_identities: Optional[
        Mapping[str, ResponsesFunctionIdentity]
    ] = None,
) -> Optional[CodexSendMessageOutputGate]:
    """Build a gate only from identities bound to the current request."""
    identities = tuple(
        identity
        for identity in get_bound_codex_collaboration_tool_identities(request)
        if is_codex_collaboration_send_message_identity(identity)
    )
    if not identities:
        return None
    if upstream_to_original_identities is None:
        try:
            from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.openai_responses_body import (
                get_bound_openai_responses_wire_body,
            )

            compiled_body = get_bound_openai_responses_wire_body(request)
            rewrite = (
                None
                if compiled_body is None
                else compiled_body.function_name_rewrite
            )
            upstream_to_original_identities = (
                None if rewrite is None else rewrite.upstream_to_original_identities
            )
        except Exception:
            upstream_to_original_identities = None
    allowed = set(identities)
    reserved_names = (
        collect_codex_collaboration_advertised_tool_names(request_body)
        if isinstance(request_body, dict)
        else frozenset()
    )
    aliases = build_codex_collaboration_wire_aliases(
        identities,
        reserved_names=reserved_names,
    )
    alias_map: dict[str, ResponsesFunctionIdentity] = {}
    ambiguous_aliases: set[str] = set()

    def add_alias(name: Any, identity: Any) -> None:
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(identity, ResponsesFunctionIdentity)
            or identity not in allowed
        ):
            return
        current = alias_map.get(name)
        if current is not None and current != identity:
            ambiguous_aliases.add(name)
            alias_map.pop(name, None)
            return
        if name not in ambiguous_aliases:
            alias_map[name] = identity

    for alias in aliases:
        add_alias(alias.upstream_name, alias.original)
    for mapping in (
        upstream_to_original_identities,
        extra_upstream_to_original_identities,
    ):
        if isinstance(mapping, Mapping):
            for name, identity in mapping.items():
                add_alias(name, identity)
    if isinstance(adapted_namespace_by_name, Mapping):
        for name, namespace in adapted_namespace_by_name.items():
            if not isinstance(name, str) or not isinstance(namespace, str):
                continue
            for identity in identities:
                if (
                    identity.namespace == namespace
                    and identity.rendered_name == name
                ):
                    add_alias(name, identity)

    pair_map: dict[tuple[str, Optional[str]], ResponsesFunctionIdentity] = {}
    ambiguous_pairs: set[tuple[str, Optional[str]]] = set()

    def add_pair(
        name: Any,
        namespace: Any,
        identity: ResponsesFunctionIdentity,
    ) -> None:
        if not isinstance(name, str) or not name:
            return
        if namespace is not None and not isinstance(namespace, str):
            return
        key = (name, namespace)
        current = pair_map.get(key)
        if current is not None and current != identity:
            ambiguous_pairs.add(key)
            pair_map.pop(key, None)
            return
        if key not in ambiguous_pairs:
            pair_map[key] = identity

    for identity in identities:
        add_pair(identity.rendered_name, identity.rendered_namespace, identity)
        add_pair(identity.name, identity.namespace, identity)
    for name, identity in alias_map.items():
        if identity.namespace is not None:
            add_pair(name, identity.namespace, identity)
    resolver = _IdentityResolver(aliases=alias_map, pairs=pair_map)
    return CodexSendMessageOutputGate(resolver)
