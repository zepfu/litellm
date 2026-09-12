"""Resolve the musela Muse TUI origin URL from litellm-alpha inspect.

Operator ``musela`` hard-codes the Tailscale alpha hostname. The harness
analogue is the same ``muse --base-url`` shape aimed at the
inspect-published litellm-alpha host port (today ``4011``). The URL is
origin-only: Muse itself joins ``GET /muse-code/models`` and
``POST /responses``. Never append ``/v1`` or ``/muse-code``. Never
target ``aawm-litellm``, ``litellm-dev``, host ports ``4000`` /
``4001``, or the ``muselt`` / ``musel`` wrappers.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
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

MUSELA_WRAPPER_NAME = "musela"
MUSELT_WRAPPER_NAME = "muselt"
MUSEL_WRAPPER_NAME = "musel"
# ``musel`` is not a URL substring: it is a prefix of ``musela``.
_FORBIDDEN_URL_TOKENS = (
    ":4000",
    ":4001",
    "aawm-litellm",
    "litellm-dev",
    MUSELT_WRAPPER_NAME,
)
_FORBIDDEN_WRAPPER_NAMES = frozenset(
    {MUSELA_WRAPPER_NAME, MUSELT_WRAPPER_NAME, MUSEL_WRAPPER_NAME}
)
_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost"})


def musela_container(config: Mapping[str, Any]) -> str:
    container = resolve_container_name(
        str(config.get("default_instance") or "litellm-alpha"),
        config,
    )
    if container != "litellm-alpha":
        raise PlanError(
            f"{MUSELA_WRAPPER_NAME} must target litellm-alpha, not {container!r}"
        )
    assert_container_allowed(container, config)
    return container


def _argv_basename(token: str) -> str:
    return Path(str(token or "")).name


def assert_musela_base_url(url: str, config: Mapping[str, Any]) -> str:
    """Fail closed unless *url* is an alpha origin, never muselt/musel/dev/prod."""

    cleaned = str(url or "").strip()
    if not cleaned:
        raise PlanError(f"{MUSELA_WRAPPER_NAME} --base-url is empty")
    lowered = cleaned.lower()
    for token in _FORBIDDEN_URL_TOKENS:
        if token in lowered:
            raise ProtectedTargetError(
                f"{MUSELA_WRAPPER_NAME} refuses {cleaned!r}: that target is "
                "protected (aawm-litellm / litellm-dev / :4000 / :4001 / "
                "muselt / musel). Use litellm-alpha."
            )
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise PlanError(
            f"{MUSELA_WRAPPER_NAME} --base-url must be an http(s) origin, "
            f"got {cleaned!r}"
        )
    host = str(parsed.hostname or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise PlanError(
            f"{MUSELA_WRAPPER_NAME} --base-url must be inspect-derived "
            f"127.0.0.1 (not Tailscale / operator musela), got {cleaned!r}"
        )
    if parsed.port is not None:
        assert_host_port_allowed(parsed.port, config)
        if int(parsed.port) in protected_ports(config):
            raise ProtectedTargetError(
                f"{MUSELA_WRAPPER_NAME} refuses host port {parsed.port}"
            )
    if parsed.query or parsed.fragment:
        raise PlanError(
            f"{MUSELA_WRAPPER_NAME} --base-url must be origin-only, got {cleaned!r}"
        )
    path = (parsed.path or "").rstrip("/") or "/"
    if path != "/":
        raise PlanError(
            f"{MUSELA_WRAPPER_NAME} --base-url must be origin-only "
            f"(no /v1, no /muse-code), got {cleaned!r}"
        )
    assert_url_allowed(cleaned, config)
    return cleaned.rstrip("/")


def musela_base_url_from_resolved(
    resolved: ResolvedInstance,
    config: Mapping[str, Any],
) -> str:
    if resolved.container in protected_containers(config):
        raise ProtectedTargetError(
            f"{MUSELA_WRAPPER_NAME} refuses {resolved.container!r}: "
            "that container is protected. Use litellm-alpha."
        )
    if resolved.container != "litellm-alpha":
        raise PlanError(
            f"{MUSELA_WRAPPER_NAME} must target litellm-alpha, not "
            f"{resolved.container!r}"
        )
    assert_host_port_allowed(resolved.host_port, config)
    url = str(resolved.base_url).rstrip("/")
    return assert_musela_base_url(url, config)


def resolve_musela_base_url(
    config: Mapping[str, Any],
    *,
    inspect_payload: Mapping[str, Any] | None = None,
    resolved: ResolvedInstance | None = None,
) -> str:
    if resolved is None:
        container = musela_container(config)
        resolved = inspect_instance(
            container,
            config,
            inspect_payload=inspect_payload,
        )
    return musela_base_url_from_resolved(resolved, config)


def musela_child_env(
    config: Mapping[str, Any],
    extra: Mapping[str, str] | None = None,
    *,
    base_url: str,
) -> dict[str, str]:
    from hv2.envscrub import scrubbed_child_env

    overlay = {
        "MUSE_DISABLE_ULTRA_ANIMATIONS": "1",
    }
    if extra:
        overlay.update({str(key): str(value) for key, value in extra.items()})
    env = scrubbed_child_env(config, overlay)
    assert_musela_base_url(base_url, config)
    for key, value in list(env.items()):
        lowered = str(value).lower()
        for token in _FORBIDDEN_URL_TOKENS:
            if token in lowered:
                raise ProtectedTargetError(
                    f"{MUSELA_WRAPPER_NAME} child env {key} refuses {token}: "
                    "never inherit muselt/musel / :4000 / :4001 / "
                    "aawm-litellm / litellm-dev"
                )
    return env


def muse_binary() -> str:
    configured = str(os.environ.get("AAWM_MUSE_REAL_BIN") or "").strip()
    if configured:
        name = _argv_basename(configured)
        if name in _FORBIDDEN_WRAPPER_NAMES:
            raise PlanError(
                "AAWM_MUSE_REAL_BIN must be the real muse binary, not "
                f"{name}; do not exec musela/muselt/musel"
            )
        return configured
    found = shutil.which("muse")
    if found:
        name = _argv_basename(found)
        if name in _FORBIDDEN_WRAPPER_NAMES:
            raise PlanError(
                f"resolved muse executable is {name}; set AAWM_MUSE_REAL_BIN "
                "to the real muse binary"
            )
        return found
    raise PlanError(
        "Muse executable is unavailable; set AAWM_MUSE_REAL_BIN to its absolute path."
    )


def musela_argv(
    args: Sequence[str],
    *,
    binary: str | None = None,
    base_url: str | None = None,
) -> list[str]:
    muse = binary or muse_binary()
    rest = [str(item) for item in args]
    if base_url:
        if "--base-url" in rest:
            index = rest.index("--base-url")
            if index + 1 >= len(rest):
                raise PlanError("muse --base-url is missing its URL argument")
            rest[index + 1] = base_url
        else:
            rest = ["--base-url", base_url, *rest]
    argv = [muse, *rest]
    for token in argv:
        if _argv_basename(token) in {MUSELT_WRAPPER_NAME, MUSEL_WRAPPER_NAME}:
            raise PlanError(
                "muse argv must not invoke muselt/musel; musela targets "
                "litellm-alpha only"
            )
    return argv
