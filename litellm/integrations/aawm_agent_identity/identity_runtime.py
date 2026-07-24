"""Runtime/client-identity (user-agent, client IP) helpers for AAWM.

Behavior-preserving Wave A2 extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so most module-level imports are intentionally absent here.
"""

# ruff: noqa: F821 - free names resolve via host globals after rebind
from __future__ import annotations


def _parse_client_identity_from_user_agent(
    user_agent: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if not user_agent:
        return None, None

    known_patterns = (
        (re.compile(r"\bclaude-code/(?P<version>[A-Za-z0-9.+_-]+)"), "claude-code"),
        (re.compile(r"\bcodex-tui/(?P<version>[A-Za-z0-9.+_-]+)"), "codex-tui"),
        (
            re.compile(r"\bGeminiCLI(?:-tui)?/(?P<version>[A-Za-z0-9.+_-]+)"),
            "gemini-cli",
        ),
        (re.compile(r"\bOpenAI/Python\s+(?P<version>[A-Za-z0-9.+_-]+)"), "openai-python"),
        (re.compile(r"\bAnthropic/Python\s+(?P<version>[A-Za-z0-9.+_-]+)"), "anthropic-python"),
    )
    for pattern, client_name in known_patterns:
        match = pattern.search(user_agent)
        if match:
            return client_name, match.group("version")

    for pattern in (_USER_AGENT_PRODUCT_RE, _USER_AGENT_PAREN_PRODUCT_RE):
        match = pattern.search(user_agent)
        if match:
            return match.group("name"), match.group("version")

    return None, None


def _extract_claude_code_version_from_metadata(
    metadata: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    billing_header_fields = metadata.get("anthropic_billing_header_fields")
    if not isinstance(billing_header_fields, dict):
        billing_header_fields = {}
    return (
        _first_non_empty_string(metadata.get("cc_version"), billing_header_fields.get("cc_version")),
        _first_non_empty_string(metadata.get("cc_entrypoint"), billing_header_fields.get("cc_entrypoint")),
    )


def _clean_session_history_client_ip_candidate(value: Any) -> Optional[str]:
    cleaned = _clean_non_empty_string(value)
    if not cleaned:
        return None
    if "," in cleaned:
        cleaned = cleaned.split(",", 1)[0].strip()
    return cleaned or None


def _canonical_session_history_client_ip(value: Any) -> Optional[str]:
    cleaned = _clean_session_history_client_ip_candidate(value)
    if not cleaned:
        return None
    try:
        return str(ipaddress.ip_address(cleaned))
    except ValueError:
        if cleaned.lower() == _SESSION_HISTORY_LOOPBACK_HOST_LABEL:
            return _SESSION_HISTORY_LOOPBACK_HOST_LABEL
        return None


_HOST_FUNCTION_NAMES = (
    "_parse_client_identity_from_user_agent",
    "_extract_claude_code_version_from_metadata",
    "_clean_session_history_client_ip_candidate",
    "_canonical_session_history_client_ip",
)


from types import FunctionType as _FunctionType


def _rebind_to_host_globals(fn, host_globals):
    rebound = _FunctionType(
        fn.__code__,
        host_globals,
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=fn.__closure__,
    )
    rebound.__kwdefaults__ = fn.__kwdefaults__
    rebound.__annotations__ = getattr(fn, "__annotations__", {})
    rebound.__dict__.update(fn.__dict__)
    rebound.__module__ = __name__
    rebound.__qualname__ = fn.__qualname__
    rebound.__doc__ = fn.__doc__
    return rebound


def install(host_globals):
    """Publish this module's helpers onto the identity host namespace.

    Plain functions are rebound so their ``__globals__`` is the identity
    package dict (record.py contract) -- free-name lookups then resolve
    through the identity namespace and monkeypatches on it stay effective.
    ``functools.lru_cache`` wrappers keep this module's globals (their bodies
    only reference module-local names) and are published by reference so the
    facade-identity invariant holds.
    """
    mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _original = mod[_name]
        if isinstance(_original, _FunctionType):
            _installed = _rebind_to_host_globals(_original, host_globals)
            mod[_name] = _installed
            host_globals[_name] = _installed
        else:
            host_globals[_name] = _original
