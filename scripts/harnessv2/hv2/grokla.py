"""Resolve the grokla Grok CLI chat-proxy URL from litellm-alpha inspect.

groklt hard-codes the litellm-dev ``:4001`` proxy. grokla is the same
``GROK_CLI_CHAT_PROXY_BASE_URL`` / ``/grok/v1`` shape aimed at the
inspect-published litellm-alpha host port (today ``4011``). It never
targets ``aawm-litellm``, ``litellm-dev``, or host ports ``4000`` /
``4001``.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from hv2.docker_guard import (
    assert_container_allowed,
    assert_host_port_allowed,
    assert_url_allowed,
    protected_containers,
    protected_ports,
)
from hv2.errors import PlanError, ProtectedTargetError
from hv2.instance import ResolvedInstance, inspect_instance, resolve_container_name

GROKLA_WRAPPER_NAME = "grokla"
GROKLT_WRAPPER_NAME = "groklt"
GROKLA_PROXY_PATH = "/grok/v1"
_FORBIDDEN_URL_TOKENS = (
    ":4000",
    ":4001",
    "aawm-litellm",
    "litellm-dev",
    GROKLT_WRAPPER_NAME,
)


def grokla_container(config: Mapping[str, Any]) -> str:
    container = resolve_container_name(
        str(config.get("default_instance") or "litellm-alpha"),
        config,
    )
    if container != "litellm-alpha":
        raise PlanError(
            f"{GROKLA_WRAPPER_NAME} must target litellm-alpha, not {container!r}"
        )
    assert_container_allowed(container, config)
    return container


def assert_grokla_proxy_url(url: str, config: Mapping[str, Any]) -> str:
    """Fail closed unless *url* is an alpha ``/grok/v1`` proxy, never groklt/dev/prod."""

    cleaned = str(url or "").strip()
    if not cleaned:
        raise PlanError(f"{GROKLA_WRAPPER_NAME} proxy URL is empty")
    lowered = cleaned.lower()
    for token in _FORBIDDEN_URL_TOKENS:
        if token in lowered:
            raise ProtectedTargetError(
                f"{GROKLA_WRAPPER_NAME} refuses {cleaned!r}: that target is "
                "protected (aawm-litellm / litellm-dev / :4000 / :4001 / groklt). "
                "Use litellm-alpha."
            )
    parsed = urlparse(cleaned)
    if parsed.port is not None:
        assert_host_port_allowed(parsed.port, config)
        if int(parsed.port) in protected_ports(config):
            raise ProtectedTargetError(
                f"{GROKLA_WRAPPER_NAME} refuses host port {parsed.port}"
            )
    path = (parsed.path or "").rstrip("/") or "/"
    if path != GROKLA_PROXY_PATH:
        raise PlanError(
            f"{GROKLA_WRAPPER_NAME} proxy URL must end with {GROKLA_PROXY_PATH}, "
            f"got {cleaned!r}"
        )
    assert_url_allowed(cleaned, config)
    return cleaned


def grokla_proxy_url_from_resolved(
    resolved: ResolvedInstance,
    config: Mapping[str, Any],
) -> str:
    if resolved.container in protected_containers(config):
        raise ProtectedTargetError(
            f"{GROKLA_WRAPPER_NAME} refuses {resolved.container!r}: "
            "that container is protected. Use litellm-alpha."
        )
    if resolved.container != "litellm-alpha":
        raise PlanError(
            f"{GROKLA_WRAPPER_NAME} must target litellm-alpha, not "
            f"{resolved.container!r}"
        )
    assert_host_port_allowed(resolved.host_port, config)
    url = f"{str(resolved.base_url).rstrip('/')}{GROKLA_PROXY_PATH}"
    return assert_grokla_proxy_url(url, config)


def resolve_grokla_proxy_url(
    config: Mapping[str, Any],
    *,
    inspect_payload: Mapping[str, Any] | None = None,
    resolved: ResolvedInstance | None = None,
) -> str:
    if resolved is None:
        container = grokla_container(config)
        resolved = inspect_instance(
            container,
            config,
            inspect_payload=inspect_payload,
        )
    return grokla_proxy_url_from_resolved(resolved, config)


def grokla_child_env(
    config: Mapping[str, Any],
    extra: Mapping[str, str] | None = None,
    *,
    proxy_url: str,
) -> dict[str, str]:
    from hv2.envscrub import scrubbed_child_env

    overlay = {
        "GROK_CLI_CHAT_PROXY_BASE_URL": assert_grokla_proxy_url(proxy_url, config),
        "GROK_DISABLE_UPDATE_CHECK": "1",
        "GROK_SANDBOX": "workspace",
        "GROK_SUBAGENTS": "1",
    }
    if extra:
        overlay.update({str(key): str(value) for key, value in extra.items()})
    return scrubbed_child_env(config, overlay)


def grok_binary() -> str:
    configured = str(os.environ.get("AAWM_GROK_REAL_BIN") or "").strip()
    if configured:
        return configured
    found = shutil.which("grok")
    if found:
        return found
    raise PlanError(
        "Grok executable is unavailable; set AAWM_GROK_REAL_BIN to its absolute path."
    )


def grokla_argv(args: Sequence[str], *, binary: str | None = None) -> list[str]:
    grok = binary or grok_binary()
    return [grok, *[str(item) for item in args]]
