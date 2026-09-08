"""Request-local function-name rewriting for OpenAI Responses payloads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional

ALGORITHM_VERSION = "responses-function-name-v1"
DEFAULT_MAX_FUNCTION_NAME_LENGTH = 64
_DIGEST_LENGTH = 16
_PRIMARY_ATTEMPTS = 32
_FALLBACK_ATTEMPTS = 32


@dataclass(frozen=True)
class ResponsesFunctionIdentity:
    """Matching identity plus the source spelling, when it was qualified."""

    name: str
    namespace: Optional[str] = None
    original_name: Optional[str] = None
    original_namespace: Optional[str] = None

    @property
    def rendered_name(self) -> str:
        """Return the name spelling that was present in the source body."""
        return (
            self.original_name
            if self.original_name is not None
            else self.name
        )

    @property
    def rendered_namespace(self) -> Optional[str]:
        """Return the source namespace, preserving dotted-name absence."""
        if self.original_name is not None:
            return self.original_namespace
        return self.namespace


@dataclass(frozen=True)
class ResponsesFunctionNameDiagnostics:
    """Bounded diagnostics that never contain function names or mappings."""

    algorithm_version: str
    max_length: int
    distinct_rewritten_count: int
    rewritten_occurrence_count: int
    affected_surfaces: tuple[str, ...]
    collision_fallback_used: bool


@dataclass(frozen=True)
class ResponsesFunctionNameRewrite:
    """Immutable request-local rewrite state and sanitized body."""

    body: Any
    original_to_upstream: Mapping[str, str]
    upstream_to_original: Mapping[str, str]
    original_identity_to_upstream: Mapping[ResponsesFunctionIdentity, str]
    upstream_to_original_identities: Mapping[str, ResponsesFunctionIdentity]
    diagnostics: ResponsesFunctionNameDiagnostics

    @property
    def changed(self) -> bool:
        return bool(
            self.original_to_upstream
            or self.original_identity_to_upstream
        )

    def restore_name(self, name: Any) -> Any:
        if not isinstance(name, str):
            return name
        identity = self.upstream_to_original_identities.get(name)
        if identity is not None:
            return identity.rendered_name
        return self.upstream_to_original.get(name, name)

    def restore_identity(
        self,
        name: Any,
        namespace: Any = None,
    ) -> tuple[Any, Any]:
        """Restore a name/namespace pair without interpreting arguments."""
        if not isinstance(name, str):
            return name, namespace
        identity = self.upstream_to_original_identities.get(name)
        if identity is not None:
            return identity.rendered_name, identity.rendered_namespace
        return self.upstream_to_original.get(name, name), namespace


def _empty_rewrite(body: Any, *, max_length: int) -> ResponsesFunctionNameRewrite:
    return ResponsesFunctionNameRewrite(
        body=body,
        original_to_upstream=MappingProxyType({}),
        upstream_to_original=MappingProxyType({}),
        original_identity_to_upstream=MappingProxyType({}),
        upstream_to_original_identities=MappingProxyType({}),
        diagnostics=ResponsesFunctionNameDiagnostics(
            algorithm_version=ALGORITHM_VERSION,
            max_length=max_length,
            distinct_rewritten_count=0,
            rewritten_occurrence_count=0,
            affected_surfaces=(),
            collision_fallback_used=False,
        ),
    )


def _validate_max_length(max_length: int) -> None:
    minimum = len("__") + _DIGEST_LENGTH + 1
    if max_length < minimum:
        raise ValueError(
            f"max_length must be at least {minimum} for deterministic rewriting"
        )


def _build_sanitized_name_candidate(
    original: str,
    max_length: int,
    *,
    nonce: int = 0,
) -> str:
    """Build a stable readable candidate no longer than ``max_length``."""
    digest_source = original if nonce == 0 else f"{original}\0{nonce}"
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
    prefix_length = max_length - len("__") - len(digest)
    return f"{original[:prefix_length]}__{digest}"


def _build_collision_fallback_candidate(
    original: str,
    *,
    ordinal: int,
    max_length: int,
    nonce: int,
) -> str:
    digest = hashlib.sha256(
        f"{ordinal}\0{original}\0{nonce}".encode("utf-8")
    ).hexdigest()[:_DIGEST_LENGTH]
    prefix = f"litellm_fn_{ordinal}_"
    available = max_length - len(prefix)
    if available <= 0:
        return digest[:max_length]
    return f"{prefix}{digest[:available]}"


def _collect_function_names(
    body: dict[str, Any],
) -> tuple[dict[str, list[str]], list[str]]:
    surfaces: dict[str, list[str]] = {
        "input": [],
        "tools": [],
        "functions": [],
        "tool_choice": [],
    }

    input_items = body.get("input")
    if isinstance(input_items, list):
        for item in input_items:
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            name = item.get("name")
            if isinstance(name, str):
                surfaces["input"].append(name)

    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            function = tool.get("function")
            name = (
                function.get("name")
                if isinstance(function, dict)
                else tool.get("name")
            )
            if isinstance(name, str):
                surfaces["tools"].append(name)

    functions = body.get("functions")
    if isinstance(functions, list):
        for function in functions:
            if not isinstance(function, dict):
                continue
            function_body = function.get("function")
            name = (
                function_body.get("name")
                if isinstance(function_body, dict)
                else function.get("name")
            )
            if isinstance(name, str):
                surfaces["functions"].append(name)

    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice.get("name")
        if isinstance(name, str):
            surfaces["tool_choice"].append(name)

    all_names = sorted({name for names in surfaces.values() for name in names})
    return surfaces, all_names


def _allocate_name_mapping(
    names: list[str],
    *,
    max_length: int,
    reserved_names: Optional[set[str] | frozenset[str]] = None,
) -> tuple[dict[str, str], bool]:
    # Deterministic allocation for tool identity / prompt-cache stability:
    # 1. Prefer the nonce=0 candidate derived from the original name alone so a
    #    tool keeps the same upstream name across requests whenever that
    #    preferred candidate is free.
    # 2. On in-request collisions (preferred candidate taken by a short name or
    #    another long name), walk nonces in order and assign in sorted-original
    #    order. Sorting — not set/dict iteration order — is the only tie-break,
    #    so the same set of originals always yields the same mapping regardless
    #    of tools[] / input[] order in the request body.
    unique_names = sorted(set(names))
    short_names = {name for name in unique_names if len(name) <= max_length}
    long_names = [name for name in unique_names if len(name) > max_length]
    used_names = set(short_names)
    used_names.update(reserved_names or ())
    mapping: dict[str, str] = {}
    collision_fallback_used = False

    for ordinal, original in enumerate(long_names):
        selected: str | None = None
        for nonce in range(_PRIMARY_ATTEMPTS):
            candidate = _build_sanitized_name_candidate(
                original,
                max_length,
                nonce=nonce,
            )
            if candidate not in used_names:
                selected = candidate
                collision_fallback_used = collision_fallback_used or nonce > 0
                break
            collision_fallback_used = True

        if selected is None:
            for nonce in range(_FALLBACK_ATTEMPTS):
                candidate = _build_collision_fallback_candidate(
                    original,
                    ordinal=ordinal,
                    max_length=max_length,
                    nonce=nonce,
                )
                if candidate not in used_names:
                    selected = candidate
                    collision_fallback_used = True
                    break

        if selected is None:
            raise ValueError(
                "Unable to allocate a unique Responses function name within "
                f"{max_length} characters"
            )

        mapping[original] = selected
        used_names.add(selected)

    return mapping, collision_fallback_used


def _rewrite_named_items(
    items: Any,
    *,
    item_type: str,
    mapping: Mapping[str, str],
    allow_legacy_function: bool = False,
) -> tuple[Any, int]:
    if not isinstance(items, list):
        return items, 0

    updated_items: list[Any] | None = None
    rewritten_count = 0
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if item_type == "function" and item.get("type") == "function":
            function = item.get("function")
            original = (
                function.get("name")
                if isinstance(function, dict)
                else item.get("name")
            )
        elif (
            allow_legacy_function
            and item_type == "function"
            and item.get("type") is None
        ):
            function = item.get("function")
            original = (
                function.get("name")
                if isinstance(function, dict)
                else item.get("name")
            )
        elif item.get("type") == item_type:
            original = item.get("name")
        else:
            continue
        if not isinstance(original, str):
            continue
        upstream = mapping.get(original)
        if upstream is None:
            continue
        if updated_items is None:
            updated_items = list(items)
        updated_item = dict(item)
        function = item.get("function")
        if isinstance(function, dict):
            updated_function = dict(function)
            updated_function["name"] = upstream
            updated_item["function"] = updated_function
        else:
            updated_item["name"] = upstream
        updated_items[index] = updated_item
        rewritten_count += 1

    return (updated_items if updated_items is not None else items), rewritten_count


def _resolve_forced_identity(
    *,
    name: Any,
    namespace: Any,
    rewrites: Mapping[ResponsesFunctionIdentity, str],
) -> tuple[ResponsesFunctionIdentity, str] | None:
    if not isinstance(name, str):
        return None
    if namespace is not None and not isinstance(namespace, str):
        return None

    if namespace is None and "." in name:
        qualified_namespace, qualified_name = name.rsplit(".", 1)
        candidates = [
            identity
            for identity in rewrites
            if identity.name == qualified_name
            and identity.namespace == qualified_namespace
            and identity.original_name == name
            and identity.original_namespace is None
        ]
        if len(candidates) == 1:
            identity = candidates[0]
            return identity, rewrites[identity]
        if candidates:
            return None
        candidates = [
            identity
            for identity in rewrites
            if identity.name == qualified_name
            and identity.namespace == qualified_namespace
        ]
    elif namespace is not None:
        candidates = [
            identity
            for identity in rewrites
            if identity.name == name and identity.namespace == namespace
        ]
    else:
        candidates = [
            identity
            for identity in rewrites
            if identity.name == name and identity.namespace is None
        ]
    if len(candidates) == 1:
        identity = candidates[0]
        return identity, rewrites[identity]
    return None


def _consistent_namespace(
    *values: Any,
) -> tuple[Optional[str], bool]:
    declarations = [value for value in values if value is not None]
    if any(not isinstance(value, str) for value in declarations):
        return None, True
    if len(set(declarations)) > 1:
        return None, True
    return (declarations[0] if declarations else None), False


def _rewrite_forced_tool_reference(
    reference: dict[str, Any],
    *,
    rewrites: Mapping[ResponsesFunctionIdentity, str],
) -> tuple[
    dict[str, Any],
    bool,
    Optional[ResponsesFunctionIdentity],
    Optional[str],
]:
    """Rewrite one supported tool-choice reference with the exact alias map."""
    nested_function = reference.get("function")
    if isinstance(nested_function, dict):
        name = nested_function.get("name")
        namespace, namespace_conflict = _consistent_namespace(
            nested_function.get("namespace"),
            reference.get("namespace"),
        )
    elif reference.get("type") == "function":
        name = reference.get("name")
        namespace, namespace_conflict = _consistent_namespace(
            reference.get("namespace"),
        )
    else:
        return reference, False, None, None

    if namespace_conflict:
        return reference, False, None, None
    resolved = _resolve_forced_identity(
        name=name,
        namespace=namespace,
        rewrites=rewrites,
    )
    if resolved is None:
        return reference, False, None, None

    identity, upstream_name = resolved
    normalized_reference = dict(reference)
    if isinstance(nested_function, dict):
        normalized_function = dict(nested_function)
        normalized_function["name"] = upstream_name
        normalized_function.pop("namespace", None)
        normalized_reference["function"] = normalized_function
        normalized_reference.pop("namespace", None)
    else:
        normalized_reference["name"] = upstream_name
        normalized_reference.pop("namespace", None)
    return normalized_reference, True, identity, upstream_name


def _rewrite_forced_function_tool(
    tool: dict[str, Any],
    *,
    namespace_context: Optional[str],
    allow_legacy_function: bool = False,
    rewrites: Mapping[ResponsesFunctionIdentity, str],
) -> tuple[
    dict[str, Any],
    bool,
    Optional[ResponsesFunctionIdentity],
    Optional[str],
]:
    if tool.get("type") == "namespace":
        return tool, False, None, None
    if tool.get("type") != "function" and not (
        allow_legacy_function and tool.get("type") is None
    ):
        return tool, False, None, None
    function = tool.get("function")
    function_body = function if isinstance(function, dict) else tool
    namespace, namespace_conflict = _consistent_namespace(
        tool.get("namespace"),
        function.get("namespace") if isinstance(function, dict) else None,
        namespace_context,
    )
    if namespace_conflict:
        return tool, False, None, None
    resolved = _resolve_forced_identity(
        name=function_body.get("name"),
        namespace=namespace,
        rewrites=rewrites,
    )
    if resolved is None:
        return tool, False, None, None
    identity, upstream_name = resolved
    normalized_tool = dict(tool)
    if isinstance(function, dict):
        normalized_function = dict(function)
        normalized_function["name"] = upstream_name
        normalized_function.pop("namespace", None)
        normalized_tool["function"] = normalized_function
        normalized_tool.pop("namespace", None)
    else:
        normalized_tool["name"] = upstream_name
        normalized_tool.pop("namespace", None)
    return normalized_tool, True, identity, upstream_name


def _rewrite_forced_collaboration_identities(
    body: dict[str, Any],
    rewrites: Mapping[ResponsesFunctionIdentity, str],
) -> tuple[
    dict[str, Any],
    dict[ResponsesFunctionIdentity, str],
    dict[str, ResponsesFunctionIdentity],
    int,
    set[str],
]:
    if not rewrites:
        return body, {}, {}, 0, set()

    advertised_surfaces, _ = _collect_function_names(body)
    advertised_names = set(
        advertised_surfaces["tools"] + advertised_surfaces["functions"]
    )
    if any(alias in advertised_names for alias in rewrites.values()):
        raise ValueError("collaboration wire alias collides with advertised tool")

    updated_body = dict(body)
    changed = False
    occurrence_count = 0
    affected_surfaces: set[str] = set()
    upstream_to_original: dict[str, ResponsesFunctionIdentity] = {}

    def record(
        identity: ResponsesFunctionIdentity,
        upstream_name: str,
        surface: str,
    ) -> None:
        nonlocal occurrence_count
        existing = upstream_to_original.get(upstream_name)
        if existing is not None and existing != identity:
            raise ValueError(
                "collaboration wire alias maps to multiple function identities"
            )
        upstream_to_original[upstream_name] = identity
        occurrence_count += 1
        affected_surfaces.add(surface)

    input_items = body.get("input")
    if isinstance(input_items, list):
        updated_input = list(input_items)
        input_changed = False
        for index, item in enumerate(input_items):
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            resolved = _resolve_forced_identity(
                name=item.get("name"),
                namespace=item.get("namespace"),
                rewrites=rewrites,
            )
            if resolved is None:
                continue
            identity, upstream_name = resolved
            updated_item = dict(item)
            updated_item["name"] = upstream_name
            updated_item.pop("namespace", None)
            updated_input[index] = updated_item
            input_changed = True
            record(identity, upstream_name, "input")
        if input_changed:
            updated_body["input"] = updated_input
            changed = True

    tools = body.get("tools")
    if isinstance(tools, list):
        updated_tools: list[Any] = []
        tools_changed = False
        for tool in tools:
            if not isinstance(tool, dict):
                updated_tools.append(tool)
                continue
            if tool.get("type") == "namespace":
                namespace = tool.get("name")
                children = tool.get("tools")
                if not isinstance(namespace, str) or not isinstance(children, list):
                    updated_tools.append(tool)
                    continue
                remaining_children: list[Any] = []
                namespace_changed = False
                for child in children:
                    if not isinstance(child, dict):
                        remaining_children.append(child)
                        continue
                    (
                        rewritten_child,
                        child_changed,
                        identity,
                        upstream_name,
                    ) = (
                        _rewrite_forced_function_tool(
                            child,
                            namespace_context=namespace,
                            rewrites=rewrites,
                        )
                    )
                    if not child_changed:
                        remaining_children.append(child)
                        continue
                    namespace_changed = True
                    tools_changed = True
                    if identity is not None:
                        assert upstream_name is not None
                        record(identity, upstream_name, "tools")
                    updated_tools.append(rewritten_child)
                if not namespace_changed:
                    updated_tools.append(tool)
                elif remaining_children:
                    updated_namespace = dict(tool)
                    updated_namespace["tools"] = remaining_children
                    updated_tools.append(updated_namespace)
                changed = changed or namespace_changed
                continue

            (
                rewritten_tool,
                tool_changed,
                identity,
                upstream_name,
            ) = _rewrite_forced_function_tool(
                tool,
                namespace_context=None,
                rewrites=rewrites,
            )
            updated_tools.append(rewritten_tool)
            if tool_changed:
                tools_changed = True
                changed = True
                if identity is not None:
                    assert upstream_name is not None
                    record(identity, upstream_name, "tools")
        if tools_changed:
            updated_body["tools"] = updated_tools

    functions = body.get("functions")
    if isinstance(functions, list):
        updated_functions: list[Any] = []
        functions_changed = False
        for function in functions:
            if not isinstance(function, dict):
                updated_functions.append(function)
                continue
            (
                rewritten_function,
                function_changed,
                identity,
                upstream_name,
            ) = _rewrite_forced_function_tool(
                function,
                namespace_context=None,
                allow_legacy_function=True,
                rewrites=rewrites,
            )
            updated_functions.append(rewritten_function)
            if not function_changed:
                continue
            functions_changed = True
            changed = True
            if identity is not None:
                assert upstream_name is not None
                record(identity, upstream_name, "functions")
        if functions_changed:
            updated_body["functions"] = updated_functions

    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict):
        updated_choice = dict(tool_choice)
        choice_changed = False
        (
            rewritten_choice,
            choice_item_changed,
            identity,
            upstream_name,
        ) = _rewrite_forced_tool_reference(
            tool_choice,
            rewrites=rewrites,
        )
        if choice_item_changed:
            assert identity is not None
            assert upstream_name is not None
            updated_choice = rewritten_choice
            record(identity, upstream_name, "tool_choice")
            choice_changed = True
        allowed_tools = tool_choice.get("tools")
        if isinstance(allowed_tools, list):
            updated_allowed_tools = list(allowed_tools)
            allowed_changed = False
            for index, allowed_tool in enumerate(allowed_tools):
                if not isinstance(allowed_tool, dict):
                    continue
                (
                    rewritten_allowed_tool,
                    allowed_item_changed,
                    identity,
                    upstream_name,
                ) = _rewrite_forced_tool_reference(
                    allowed_tool,
                    rewrites=rewrites,
                )
                if not allowed_item_changed:
                    continue
                assert identity is not None
                assert upstream_name is not None
                updated_allowed_tools[index] = rewritten_allowed_tool
                record(identity, upstream_name, "tool_choice")
                allowed_changed = True
            if allowed_changed:
                updated_choice["tools"] = updated_allowed_tools
                choice_changed = True
        if choice_changed:
            updated_body["tool_choice"] = updated_choice
            changed = True

    if not changed:
        return body, {}, {}, 0, set()
    return (
        updated_body,
        dict(rewrites),
        upstream_to_original,
        occurrence_count,
        affected_surfaces,
    )


def sanitize_responses_function_names(
    body: Any,
    *,
    max_length: int = DEFAULT_MAX_FUNCTION_NAME_LENGTH,
    forced_identity_rewrites: Optional[
        Mapping[ResponsesFunctionIdentity, str]
    ] = None,
) -> ResponsesFunctionNameRewrite:
    """Sanitize function names and compose scoped native identity rewrites."""
    _validate_max_length(max_length)
    if not isinstance(body, dict):
        return _empty_rewrite(body, max_length=max_length)

    (
        forced_body,
        forced_original_to_upstream,
        forced_upstream_to_original,
        forced_occurrence_count,
        forced_surfaces,
    ) = _rewrite_forced_collaboration_identities(
        body,
        forced_identity_rewrites or {},
    )
    surfaces, names = _collect_function_names(forced_body)
    mapping, collision_fallback_used = _allocate_name_mapping(
        names,
        max_length=max_length,
        reserved_names=set((forced_identity_rewrites or {}).values()),
    )
    if not mapping and not forced_original_to_upstream:
        return _empty_rewrite(body, max_length=max_length)

    updated_input, input_count = _rewrite_named_items(
        forced_body.get("input"),
        item_type="function_call",
        mapping=mapping,
    )
    updated_tools, tool_count = _rewrite_named_items(
        forced_body.get("tools"),
        item_type="function",
        mapping=mapping,
    )
    updated_functions, functions_count = _rewrite_named_items(
        forced_body.get("functions"),
        item_type="function",
        mapping=mapping,
        allow_legacy_function=True,
    )

    updated_tool_choice = forced_body.get("tool_choice")
    tool_choice_count = 0
    if (
        isinstance(updated_tool_choice, dict)
        and updated_tool_choice.get("type") == "function"
        and isinstance(updated_tool_choice.get("name"), str)
    ):
        original_choice = updated_tool_choice["name"]
        upstream_choice = mapping.get(original_choice)
        if upstream_choice is not None:
            updated_tool_choice = dict(updated_tool_choice)
            updated_tool_choice["name"] = upstream_choice
            tool_choice_count = 1

    updated_body = dict(forced_body)
    if input_count:
        updated_body["input"] = updated_input
    if tool_count:
        updated_body["tools"] = updated_tools
    if functions_count:
        updated_body["functions"] = updated_functions
    if tool_choice_count:
        updated_body["tool_choice"] = updated_tool_choice

    occurrence_count = (
        forced_occurrence_count
        + input_count
        + tool_count
        + functions_count
        + tool_choice_count
    )
    affected_surfaces = tuple(
        surface for surface in ("input", "tools", "functions", "tool_choice")
        if surface in forced_surfaces
        or any(name in mapping for name in surfaces[surface])
    )
    upstream_to_original = {
        upstream: original for original, upstream in mapping.items()
    }
    return ResponsesFunctionNameRewrite(
        body=updated_body,
        original_to_upstream=MappingProxyType(dict(mapping)),
        upstream_to_original=MappingProxyType(upstream_to_original),
        original_identity_to_upstream=MappingProxyType(
            dict(forced_original_to_upstream)
        ),
        upstream_to_original_identities=MappingProxyType(
            dict(forced_upstream_to_original)
        ),
        diagnostics=ResponsesFunctionNameDiagnostics(
            algorithm_version=ALGORITHM_VERSION,
            max_length=max_length,
            distinct_rewritten_count=(
                len(mapping) + len(forced_original_to_upstream)
            ),
            rewritten_occurrence_count=occurrence_count,
            affected_surfaces=affected_surfaces,
            collision_fallback_used=collision_fallback_used,
        ),
    )


def _restore_function_call_names(
    value: Any,
    upstream_to_original: Mapping[str, str],
    upstream_to_original_identities: Mapping[
        str, ResponsesFunctionIdentity
    ],
) -> tuple[Any, bool]:
    if isinstance(value, dict):
        changed = False
        updated: dict[str, Any] | None = None

        if value.get("type") == "function_call":
            name = value.get("name")
            namespace = value.get("namespace")
            restored_name = name
            restored_namespace = namespace
            identity = (
                upstream_to_original_identities.get(name)
                if isinstance(name, str)
                else None
            )
            if (
                identity is not None
                and namespace is not None
                and (
                    not isinstance(namespace, str)
                    or namespace != identity.namespace
                )
            ):
                identity = None
            if identity is not None:
                restored_name = identity.rendered_name
                restored_namespace = identity.rendered_namespace
            elif isinstance(name, str) and name in upstream_to_original:
                restored_name = upstream_to_original[name]
            if restored_name != name or restored_namespace != namespace:
                updated = dict(value)
                updated["name"] = restored_name
                if restored_namespace is None:
                    updated.pop("namespace", None)
                else:
                    updated["namespace"] = restored_namespace
                changed = True

        source = updated if updated is not None else value
        for key, child in source.items():
            if key in {"name", "namespace"} and changed:
                continue
            restored_child, child_changed = _restore_function_call_names(
                child,
                upstream_to_original,
                upstream_to_original_identities,
            )
            if not child_changed:
                continue
            if updated is None:
                updated = dict(value)
            updated[key] = restored_child
            changed = True

        return (updated if updated is not None else value), changed

    if isinstance(value, list):
        updated_items: list[Any] | None = None
        for index, item in enumerate(value):
            restored_item, item_changed = _restore_function_call_names(
                item,
                upstream_to_original,
                upstream_to_original_identities,
            )
            if not item_changed:
                continue
            if updated_items is None:
                updated_items = list(value)
            updated_items[index] = restored_item
        return (updated_items if updated_items is not None else value), (
            updated_items is not None
        )

    return value, False


def _restore_forced_tool_reference(
    reference: Any,
    upstream_to_original_identities: Mapping[
        str, ResponsesFunctionIdentity
    ],
    *,
    namespace_context: Optional[str] = None,
) -> tuple[Any, bool]:
    if not isinstance(reference, dict):
        return reference, False

    nested_function = reference.get("function")
    if isinstance(nested_function, dict):
        name = nested_function.get("name")
        namespace, namespace_conflict = _consistent_namespace(
            nested_function.get("namespace"),
            reference.get("namespace"),
            namespace_context,
        )
    elif reference.get("type") == "function":
        name = reference.get("name")
        namespace, namespace_conflict = _consistent_namespace(
            reference.get("namespace"),
            namespace_context,
        )
    else:
        return reference, False

    if namespace_conflict:
        return reference, False
    if not isinstance(name, str):
        return reference, False
    identity = upstream_to_original_identities.get(name)
    if identity is None:
        return reference, False
    if namespace is not None and namespace != identity.namespace:
        return reference, False

    restored = dict(reference)
    if isinstance(nested_function, dict):
        restored_function = dict(nested_function)
        restored_function["name"] = identity.rendered_name
        restored_function.pop("namespace", None)
        if identity.rendered_namespace is None:
            restored.pop("namespace", None)
        else:
            restored["namespace"] = identity.rendered_namespace
        restored["function"] = restored_function
    else:
        restored["name"] = identity.rendered_name
        if identity.rendered_namespace is None:
            restored.pop("namespace", None)
        else:
            restored["namespace"] = identity.rendered_namespace
    return restored, True


def _restore_forced_tool_identity_surfaces(
    value: Any,
    upstream_to_original_identities: Mapping[
        str, ResponsesFunctionIdentity
    ],
) -> tuple[Any, bool]:
    """Restore CFG-047 identities only on known response tool surfaces."""
    if not isinstance(value, dict):
        return value, False

    updated: dict[str, Any] | None = None

    for key in ("tools", "functions"):
        items = value.get(key)
        if not isinstance(items, list):
            continue
        restored_items: list[Any] | None = None
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "namespace":
                children = item.get("tools")
                if not isinstance(children, list):
                    continue
                restored_children: list[Any] | None = None
                for child_index, child in enumerate(children):
                    restored_child, child_changed = (
                        _restore_forced_tool_reference(
                            child,
                            upstream_to_original_identities,
                            namespace_context=item.get("name"),
                        )
                    )
                    if not child_changed:
                        continue
                    if restored_children is None:
                        restored_children = list(children)
                    restored_children[child_index] = restored_child
                if restored_children is None:
                    continue
                restored_item = dict(item)
                restored_item["tools"] = restored_children
                if restored_items is None:
                    restored_items = list(items)
                restored_items[index] = restored_item
                continue

            restored_item, item_changed = _restore_forced_tool_reference(
                item,
                upstream_to_original_identities,
            )
            if not item_changed:
                continue
            if restored_items is None:
                restored_items = list(items)
            restored_items[index] = restored_item
        if restored_items is not None:
            if updated is None:
                updated = dict(value)
            updated[key] = restored_items

    tool_choice = value.get("tool_choice")
    if isinstance(tool_choice, dict):
        restored_choice, choice_changed = _restore_forced_tool_reference(
            tool_choice,
            upstream_to_original_identities,
        )
        restored_choice = (
            restored_choice if isinstance(restored_choice, dict) else tool_choice
        )
        allowed_tools = tool_choice.get("tools")
        if isinstance(allowed_tools, list):
            restored_allowed_tools: list[Any] | None = None
            for index, allowed_tool in enumerate(allowed_tools):
                restored_allowed_tool, allowed_changed = (
                    _restore_forced_tool_reference(
                        allowed_tool,
                        upstream_to_original_identities,
                    )
                )
                if not allowed_changed:
                    continue
                if restored_allowed_tools is None:
                    restored_allowed_tools = list(allowed_tools)
                restored_allowed_tools[index] = restored_allowed_tool
            if restored_allowed_tools is not None:
                if not isinstance(restored_choice, dict):
                    restored_choice = dict(tool_choice)
                restored_choice["tools"] = restored_allowed_tools
                choice_changed = True
        if choice_changed:
            if updated is None:
                updated = dict(value)
            updated["tool_choice"] = restored_choice

    return (updated if updated is not None else value), updated is not None


def restore_function_names_in_responses_body(
    body: Any,
    upstream_to_original: Mapping[str, str],
    upstream_to_original_identities: Optional[
        Mapping[str, ResponsesFunctionIdentity]
    ] = None,
) -> Any:
    """Restore exact function-call names in a Responses body or terminal event."""
    if not upstream_to_original and not upstream_to_original_identities:
        return body
    restored, _ = _restore_function_call_names(
        body,
        upstream_to_original,
        upstream_to_original_identities or {},
    )
    if upstream_to_original_identities:
        restored, _ = _restore_forced_tool_identity_surfaces(
            restored,
            upstream_to_original_identities,
        )
    return restored


def restore_function_names_in_responses_output(
    output: Any,
    upstream_to_original: Mapping[str, str],
    upstream_to_original_identities: Optional[
        Mapping[str, ResponsesFunctionIdentity]
    ] = None,
) -> Any:
    """Restore exact function-call names in a Responses output list."""
    if not upstream_to_original and not upstream_to_original_identities:
        return output
    restored, _ = _restore_function_call_names(
        output,
        upstream_to_original,
        upstream_to_original_identities or {},
    )
    return restored
