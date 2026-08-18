#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import http.client
import json
import os
import pathlib
import re
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextvars import ContextVar
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "scripts" / "local-ci" / "config.json"
# Cap CLI stdout/stderr persisted into acceptance artifacts (RR-080 #3).
_DEFAULT_CLI_OUTPUT_MAX_CHARS = 200_000
# Child CLI env policy (RR-080 #1): never inherit harness/DB/Langfuse secrets.
_CHILD_ENV_BASE_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "COLORTERM",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "DBUS_SESSION_BUS_ADDRESS",
        "SSH_AUTH_SOCK",
        "XDG_RUNTIME_DIR",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_STATE_HOME",
        "NO_COLOR",
        "FORCE_COLOR",
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "no_proxy",
        "ALL_PROXY",
        "all_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        "NODE_OPTIONS",
        "ELECTRON_RUN_AS_NODE",
    }
)
# Provider CLI prefixes intentionally may carry auth tokens/keys when the CLI
# uses env-based credentials rather than a local credential file. That is not
# the same as LiteLLM *proxy admin* secrets (master key, salt, etc.), which must
# never be inherited by agentic child processes.
_CHILD_ENV_ALLOW_PREFIXES = (
    "ANTHROPIC_",
    "CLAUDE_",
    "CODEX_",
    "OPENAI_",
    "XAI_",
)
# Narrow non-secret LiteLLM routing / logging knobs only. Do NOT add a broad
# LITELLM_ prefix here: LITELLM_MASTER_KEY and similar bypassed substring denylists
# when the prefix was treated as fully trusted.
_CHILD_ENV_ALLOW_KEYS = frozenset(
    {
        "LITELLM_BASE_URL",
        "LITELLM_API_BASE",
        "LITELLM_LOG",
        "LITELLM_MODE",
        "LITELLM_LOCAL_MODEL_COST_MAP",
    }
)
_CHILD_ENV_DENY_KEYS = frozenset(
    {
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_QUERY_URL",
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
        "DATABASE_URL",
        "DIRECT_URL",
        "PRISMA_DATABASE_URL",
        "POSTGRES_PASSWORD",
        "POSTGRES_USER",
        "POSTGRES_DB",
        "PGPASSWORD",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SESSION_TOKEN",
        "LITELLM_MASTER_KEY",
        "LITELLM_SALT_KEY",
        "LITELLM_SALT",
    }
)
_CHILD_ENV_DENY_PREFIXES = (
    "LANGFUSE_",
    "AAWM_DB_",
    "DATABASE_",
    "POSTGRES_",
    "PG",
)
_CHILD_ENV_DENY_SUBSTRINGS = (
    "SECRET",
    "PASSWORD",
    "TOKEN",
    "CREDENTIAL",
    "PRIVATE_KEY",
)
_CLAUDE_AGENT_NAME_RE = re.compile(r"You are '([^']+)' and you are working")
_CLAUDE_HARNESS_HEADER_KEYS = {
    "x-litellm-end-user-id",
    "langfuse_trace_user_id",
    "langfuse_trace_name",
}
_HARNESS_USER_ID_ENV_KEYS = (
    "AAWM_HARNESS_USER_ID",
    "PYTEST_CLASSIFIER_HARNESS_USER_ID",
    "AAWM_CLAUDE_HARNESS_USER_ID",
)
_PYTEST_CLASSIFIER_SERVICE_RE = re.compile(r"(^|[-_.])pytest-classifier($|[-_.])")


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _isoformat(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _cli_output_max_chars() -> int:
    raw = os.environ.get("ACCEPTANCE_CLI_OUTPUT_MAX_CHARS")
    if raw is None or not str(raw).strip():
        return _DEFAULT_CLI_OUTPUT_MAX_CHARS
    try:
        value = int(str(raw).strip())
    except ValueError:
        return _DEFAULT_CLI_OUTPUT_MAX_CHARS
    return max(0, value)


def _truncate_captured_text(text: str, max_chars: int | None = None) -> tuple[str, bool]:
    """Return (possibly truncated text, was_truncated)."""
    if max_chars is None:
        max_chars = _cli_output_max_chars()
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    omitted = len(text) - max_chars
    marker = f"\n...[truncated {omitted} chars; original_len={len(text)}]...\n"
    # Prefer keeping the head of the stream (session ids / JSON often appear early).
    keep = max_chars - len(marker)
    if keep <= 0:
        return marker[:max_chars], True
    return text[:keep] + marker, True


def _is_denied_child_env_key(key: str) -> bool:
    """Deny harness/Langfuse/DB/LiteLLM-admin secrets; provider CLI auth may pass.

    Distinction:
    - Provider prefixes (ANTHROPIC_/OPENAI_/CODEX_/…): intentional allow of
      env-based CLI credentials (API keys/tokens) so codex/claude can
      authenticate without a file-based login when the operator has only env auth.
    - LITELLM_*: default-deny except the narrow non-secret routing allowlist in
      ``_CHILD_ENV_ALLOW_KEYS``. ``LITELLM_MASTER_KEY`` and other proxy admin
      material must never reach agentic child CLIs (tools can dump ``env``).
    - Langfuse/DB/Postgres and generic SECRET/PASSWORD/TOKEN names: always deny.
    """
    if key in _CHILD_ENV_DENY_KEYS:
        return True
    for prefix in _CHILD_ENV_DENY_PREFIXES:
        if key.startswith(prefix):
            return True
    # Default-deny all LiteLLM proxy vars except the explicit non-secret set.
    if key.startswith("LITELLM_"):
        return key not in _CHILD_ENV_ALLOW_KEYS
    # Provider-prefixed vars may include API keys/tokens required by CLIs.
    for prefix in _CHILD_ENV_ALLOW_PREFIXES:
        if key.startswith(prefix):
            return False
    upper = key.upper()
    for fragment in _CHILD_ENV_DENY_SUBSTRINGS:
        if fragment in upper:
            return True
    return False


def _is_allowed_child_env_key(key: str) -> bool:
    if _is_denied_child_env_key(key):
        return False
    if key in _CHILD_ENV_BASE_KEYS or key in _CHILD_ENV_ALLOW_KEYS:
        return True
    for prefix in _CHILD_ENV_ALLOW_PREFIXES:
        if key.startswith(prefix):
            return True
    return False


def _scrubbed_child_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build a minimal env for provider CLIs without harness/DB/Langfuse secrets."""
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if value is None:
            continue
        if _is_allowed_child_env_key(key):
            env[key] = value
    if extra_env:
        for key, value in extra_env.items():
            if value is None:
                continue
            key_str = str(key)
            if _is_denied_child_env_key(key_str):
                continue
            env[key_str] = str(value)
    return env


def _expand_at_path_token(token: str, config_dir: pathlib.Path) -> str:
    """Expand ``@{config_dir}/...`` and relative ``@path`` against config_dir.

    Absolute ``@/path`` tokens are left unchanged. Coordinates with the portable
    harness bundle packaging (RR-077/RR-079).
    """
    if not isinstance(token, str) or not token.startswith("@"):
        return token
    path_part = token[1:]
    if not path_part:
        return token
    if path_part.startswith("{config_dir}/") or path_part.startswith("{config_dir}\\"):
        relative = path_part[len("{config_dir}/") :]
        return "@" + str((config_dir / relative).resolve())
    candidate = pathlib.Path(path_part)
    if candidate.is_absolute():
        return token
    # Relative @path → resolve against the config file directory.
    return "@" + str((config_dir / path_part).resolve())


def _rewrite_config_path_tokens(value: Any, config_dir: pathlib.Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_config_path_tokens(item, config_dir)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_config_path_tokens(item, config_dir) for item in value]
    if isinstance(value, str):
        return _expand_at_path_token(value, config_dir)
    return value


def _load_suite_config(path: pathlib.Path) -> dict[str, Any]:
    config = _load_json(path)
    if not isinstance(config, dict):
        raise SystemExit("Acceptance config must be a JSON object")
    return _rewrite_config_path_tokens(config, path.resolve().parent)


def _generation_quality_flags(config: dict[str, Any]) -> tuple[bool, bool]:
    """Return (skip_quality_checks, allow_zero_cost) from per-family config."""
    skip_quality = bool(
        config.get("skip_generation_quality_checks")
        or config.get("skip_quality_checks")
    )
    allow_zero_cost = bool(config.get("allow_zero_cost"))
    return skip_quality, allow_zero_cost


def _enforce_minimum_trace_count(
    *,
    family: str,
    traces: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    raw = config.get("minimum_trace_count")
    if raw is None:
        return []
    try:
        minimum = int(raw)
    except (TypeError, ValueError):
        return [f"{family} minimum_trace_count is not an integer: {raw!r}"]
    if minimum <= 0:
        return []
    actual = len(traces)
    if actual < minimum:
        return [
            f"{family} trace count {actual} < minimum_trace_count {minimum}"
        ]
    return []


def _resolve_litellm_base_url(config: dict[str, Any]) -> str:
    return os.environ.get("LITELLM_BASE_URL") or config.get(
        "litellm_base_url", "http://127.0.0.1:4001"
    )


# ---------------------------------------------------------------------------
# Target profile system (D1-574/MS-033): consistent dev/prod route rewriting.
# Follows the established adapter-harness pattern
# (run_anthropic_adapter_acceptance.py).
# ---------------------------------------------------------------------------

BUILT_IN_TARGET_PROFILES: dict[str, dict[str, str]] = {
    "dev": {
        "litellm_base_url": "http://127.0.0.1:4001",
        "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        "codex_profile": "litellm-dev",
        "docker_container_name": "litellm-dev",
        "expected_trace_environment": "dev",
    },
    "prod": {
        "litellm_base_url": "http://127.0.0.1:4000",
        "anthropic_base_url": "http://127.0.0.1:4000/anthropic",
        "codex_profile": "litellm",
        "docker_container_name": "aawm-litellm",
        "expected_trace_environment": "prod",
    },
}

_DOCKER_SUBPROCESS_TIMEOUT_SECONDS = 15
_TARGET_PROFILE_REQUIRED_KEYS = (
    "litellm_base_url",
    "anthropic_base_url",
    "codex_profile",
    "docker_container_name",
    "expected_trace_environment",
)
_TARGET_PROFILE_ARTIFACT_KEYS = (
    "target_name",
    *_TARGET_PROFILE_REQUIRED_KEYS,
)


def _resolve_container_env_value(
    container_name: str,
    env_name: str,
    *,
    cache: dict[tuple[str, str], str] | None = None,
) -> str | None:
    """Retrieve a named env var from a running container via docker exec.

    Successful results may be cached within one harness invocation. Failed
    lookups are never cached. Values are never printed, logged, or persisted.
    """
    cache_key = (container_name, env_name)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    value: str | None = None
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "printenv", env_name],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
            timeout=_DOCKER_SUBPROCESS_TIMEOUT_SECONDS,
        )
        if result.returncode == 0 and result.stdout.strip():
            value = result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    if value is not None and cache is not None:
        cache[cache_key] = value
    return value


def _merged_target_profiles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = {name: dict(profile) for name, profile in BUILT_IN_TARGET_PROFILES.items()}
    if "target_profiles" in config and not isinstance(config["target_profiles"], dict):
        raise SystemExit("target_profiles must be an object")
    configured = config.get("target_profiles")
    if isinstance(configured, dict):
        for name, configured_profile in configured.items():
            if not isinstance(name, str) or not isinstance(configured_profile, dict):
                raise SystemExit("target_profiles must map names to profile objects")
            profile = dict(profiles.get(name, {}))
            if not all(isinstance(key, str) for key in configured_profile):
                raise SystemExit("target profile keys must be strings")
            profile.update(configured_profile)
            profiles[name] = profile
    return profiles


def _validate_target_profile(name: str, profile: dict[str, Any]) -> None:
    invalid = [
        key
        for key in _TARGET_PROFILE_REQUIRED_KEYS
        if not isinstance(profile.get(key), str) or not profile[key].strip()
    ]
    if invalid:
        raise SystemExit(
            f"Acceptance target `{name}` has missing or non-string required keys: "
            f"{', '.join(invalid)}"
        )


def _normalize_base_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _infer_target_from_base_url(
    config: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
) -> str:
    """Infer exactly one merged target from the effective LiteLLM base URL.

    Unknown and ambiguous URLs fail closed before lifecycle or client work.
    """
    base_url = _normalize_base_url(_resolve_litellm_base_url(config))
    matches = [
        name
        for name, profile in profiles.items()
        if _normalize_base_url(profile.get("litellm_base_url")) == base_url
    ]
    if not matches:
        raise SystemExit(
            f"Effective LITELLM_BASE_URL `{base_url}` does not match any configured "
            "acceptance target profile"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"Effective LITELLM_BASE_URL `{base_url}` is ambiguous across target "
            f"profiles: {', '.join(sorted(matches))}"
        )
    return matches[0]


def _resolve_target_profile(
    config: dict[str, Any], target: str | None = None
) -> dict[str, str]:
    """Resolve the effective target profile from CLI arg, env, config, or URL.

    Precedence: explicit ``target`` arg > ``ACCEPTANCE_TARGET`` env >
    config ``target`` key > inferred merged profile match for the effective
    ``LITELLM_BASE_URL``. Explicit target and URL conflicts fail closed.
    """
    profiles = _merged_target_profiles(config)
    env_target = os.environ.get("ACCEPTANCE_TARGET")
    config_target = config.get("target")
    effective_target = target or env_target or config_target
    target_source = (
        "command line"
        if target
        else "ACCEPTANCE_TARGET"
        if env_target
        else "config target"
        if config_target
        else "LITELLM_BASE_URL"
    )
    if not effective_target:
        effective_target = _infer_target_from_base_url(config, profiles)
    if effective_target not in profiles:
        valid = ", ".join(sorted(profiles))
        raise SystemExit(
            f"Unknown acceptance target `{effective_target}`. Valid targets: {valid}"
        )
    profile = dict(profiles[effective_target])
    _validate_target_profile(str(effective_target), profile)
    explicit_url = os.environ.get("LITELLM_BASE_URL")
    if explicit_url and _normalize_base_url(explicit_url) != _normalize_base_url(
        profile["litellm_base_url"]
    ):
        raise SystemExit(
            f"{target_source} selected target `{effective_target}` but "
            f"LITELLM_BASE_URL `{explicit_url}` conflicts with its profile URL "
            f"`{profile['litellm_base_url']}`"
        )
    if config_target and not target and not env_target:
        configured_url = config.get("litellm_base_url")
        if configured_url and _normalize_base_url(configured_url) != _normalize_base_url(
            profile["litellm_base_url"]
        ):
            raise SystemExit(
                f"config target `{effective_target}` conflicts with configured "
                f"litellm_base_url `{configured_url}`"
            )
    profile["target_name"] = str(effective_target)
    return profile


def _public_target_profile(profile: dict[str, str]) -> dict[str, str]:
    """Return the non-secret target fields permitted in artifacts/preflight."""
    return {
        key: profile[key]
        for key in _TARGET_PROFILE_ARTIFACT_KEYS
        if isinstance(profile.get(key), str)
    }


def _require_family_config(config: dict[str, Any], family_name: str) -> dict[str, Any]:
    family = config.get(family_name)
    if not isinstance(family, dict):
        raise SystemExit(f"Acceptance config requires `{family_name}` as an object")
    command = family.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(token, str) for token in command)
    ):
        raise SystemExit(
            f"Acceptance config `{family_name}.command` must be a non-empty list[str]"
        )
    if "env" in family and not isinstance(family["env"], dict):
        raise SystemExit(f"Acceptance config `{family_name}.env` must be an object")
    return family


def _apply_target_profile(config: dict[str, Any], profile: dict[str, str]) -> None:
    """Rewrite family routes in-place for the selected target profile.

    Codex: rewrites the ``-p <profile>`` argument.
    Claude: rewrites ``env.ANTHROPIC_BASE_URL``.
    All families: ``expected_trace_environment`` is set from the profile.
    """
    config["litellm_base_url"] = profile["litellm_base_url"]
    expected_env = profile["expected_trace_environment"]
    codex = _require_family_config(config, "codex")
    claude = _require_family_config(config, "claude")

    # --- Codex: rewrite -p <profile> in command (fail closed) ---
    codex["expected_trace_environment"] = expected_env
    codex_command = codex["command"]
    codex_profile = profile["codex_profile"]
    profile_flags = [
        i for i, token in enumerate(codex_command) if token == "-p"
    ]
    if len(profile_flags) != 1 or profile_flags[0] + 1 >= len(codex_command):
        raise SystemExit(
            f"Cannot apply target profile: codex command requires exactly one "
            f"rewritable `-p <profile>` argument. Refusing to run "
            f"with an unrouted command under target "
            f"`{profile.get('expected_trace_environment', '?')}`."
        )
    codex_command[profile_flags[0] + 1] = codex_profile

    # --- Claude: rewrite ANTHROPIC_BASE_URL ---
    claude["expected_trace_environment"] = expected_env
    env_block = claude.setdefault("env", {})
    env_block["ANTHROPIC_BASE_URL"] = profile["anthropic_base_url"]
    fanout_modes = claude.get("fanout_modes", {})
    if not isinstance(fanout_modes, dict):
        raise SystemExit(
            "Cannot apply target profile: claude fanout_modes must be an object"
        )
    for mode_name, mode_cfg in fanout_modes.items():
        if not isinstance(mode_name, str) or not isinstance(mode_cfg, dict):
            raise SystemExit(
                "Cannot apply target profile: claude fanout modes must map names "
                "to objects"
            )
        mode_command = mode_cfg.get("command")
        if (
            not isinstance(mode_command, list)
            or not mode_command
            or not all(isinstance(token, str) for token in mode_command)
        ):
            raise SystemExit(
                f"Acceptance config `claude.fanout_modes.{mode_name}.command` "
                "must be a non-empty list[str]"
            )
        mode_env = mode_cfg.setdefault("env", {})
        if not isinstance(mode_env, dict):
            raise SystemExit(
                f"Cannot apply target profile: claude fanout mode "
                f"`{mode_name}` env must be an object"
            )
        mode_cfg["expected_trace_environment"] = expected_env
        mode_env["ANTHROPIC_BASE_URL"] = profile["anthropic_base_url"]


def _resolve_langfuse_credentials(
    config: dict[str, Any],
    profile: dict[str, str] | None,
    *,
    container_env_cache: dict[tuple[str, str], str] | None = None,
) -> tuple[str, str]:
    """Resolve Langfuse credentials from the target container or host env.

    When config declares ``langfuse_credential_source=target_container`` and a
    profile with ``docker_container_name`` is active, credentials are resolved
    exclusively from the container. Missing/unreadable container credentials
    fail closed -- no fallback to host/.env values. Secret values are never
    logged or persisted.
    """
    public_key_env = config.get("langfuse_public_key_env", "LANGFUSE_PUBLIC_KEY")
    secret_key_env = config.get("langfuse_secret_key_env", "LANGFUSE_SECRET_KEY")
    credential_source = config.get("langfuse_credential_source", "host_env")

    if credential_source == "target_container":
        if profile is None:
            raise SystemExit(
                "langfuse_credential_source=target_container requires a resolved "
                "acceptance target profile"
            )
        container = profile.get("docker_container_name", "")
        if not container:
            raise SystemExit(
                "langfuse_credential_source=target_container but target profile "
                "has no docker_container_name"
            )
        pk = _resolve_container_env_value(
            container, public_key_env, cache=container_env_cache
        )
        sk = _resolve_container_env_value(
            container, secret_key_env, cache=container_env_cache
        )
        if not pk or not sk:
            raise SystemExit(
                f"Failed to resolve Langfuse credentials from container "
                f"`{container}` ({public_key_env}/{secret_key_env}). "
                f"Refusing to fall back to host environment."
            )
        return pk, sk

    # Default: host environment (existing behavior).
    pk = os.environ.get(public_key_env, "")
    sk = os.environ.get(secret_key_env, "")
    if not pk or not sk:
        raise SystemExit(
            f"Missing Langfuse credentials in env vars {public_key_env}/{secret_key_env}"
        )
    return pk, sk



def _build_claude_harness_user_id(
    *, target: str | None = None, case_name: str | None = None
) -> str:
    for env_key in _HARNESS_USER_ID_ENV_KEYS:
        override = os.environ.get(env_key)
        if override and override.strip():
            return override.strip()

    service_name = os.environ.get("AAWM_OBSERVE_SERVICE_NAME", "")
    if _PYTEST_CLASSIFIER_SERVICE_RE.search(service_name.strip()):
        return "pytest-classifier"

    if os.environ.get("PYTEST_CLASSIFIER_ENABLE_OBSERVABILITY", "").strip().lower() in {
        "1",
        "on",
        "true",
        "yes",
    }:
        return "pytest-classifier"

    run_id = os.environ.get("AAWM_HARNESS_RUN_ID")
    if not run_id or not run_id.strip():
        run_id = f"{int(time.time())}-{os.getpid()}"
    scope = ".".join(
        re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
        for value in (target or "local", case_name or "claude")
        if value and value.strip()
    )
    return f"litellm-harness.{scope}.{run_id}"


def _parse_claude_custom_header_lines(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, str) or not value.strip():
        return []
    headers: list[tuple[str, str]] = []
    for line in value.splitlines():
        if ":" not in line:
            continue
        key, header_value = line.split(":", 1)
        key = key.strip()
        header_value = header_value.strip()
        if key and header_value:
            headers.append((key, header_value))
    return headers


def _format_claude_custom_header_lines(headers: list[tuple[str, str]]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in headers)


def _ensure_claude_harness_headers(
    config: dict[str, Any],
    *,
    target: str | None = None,
    case_name: str | None = None,
) -> dict[str, Any]:
    updated = dict(config)
    env = dict(updated.get("env") or {})
    expected_user_ids = [
        str(value).strip()
        for value in (updated.get("expected_user_ids") or [])
        if isinstance(value, (str, int, float)) and str(value).strip()
    ]
    harness_user_id = (
        expected_user_ids[0]
        if expected_user_ids
        else _build_claude_harness_user_id(target=target, case_name=case_name)
    )

    existing_headers = _parse_claude_custom_header_lines(
        env.get("ANTHROPIC_CUSTOM_HEADERS")
    )
    existing_trace_name = next(
        (
            value
            for key, value in existing_headers
            if key.lower() == "langfuse_trace_name"
        ),
        "claude-code",
    )
    controlled_headers = [
        ("x-litellm-end-user-id", harness_user_id),
        ("langfuse_trace_user_id", harness_user_id),
        ("langfuse_trace_name", existing_trace_name),
    ]
    passthrough_headers = [
        (key, value)
        for key, value in existing_headers
        if key.lower() not in _CLAUDE_HARNESS_HEADER_KEYS
    ]
    env["ANTHROPIC_CUSTOM_HEADERS"] = _format_claude_custom_header_lines(
        controlled_headers + passthrough_headers
    )
    updated["env"] = env
    updated["expected_user_ids"] = expected_user_ids or [harness_user_id]
    updated.setdefault("require_trace_user_id", True)
    return updated


def _http_get_json(
    url: str,
    public_key: str,
    secret_key: str,
    timeout: float = 20.0,
    *,
    deadline: float | None = None,
) -> dict[str, Any]:
    credentials = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Basic {credentials}",
            "Accept": "application/json",
        },
        method="GET",
    )
    last_error: Exception | None = None

    def bounded_timeout() -> float:
        if deadline is None:
            return timeout
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError("Langfuse lookup deadline exceeded")
        return min(timeout, remaining_seconds)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(
                request,
                timeout=bounded_timeout(),
            ) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except (
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            ConnectionResetError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt < 2:
                retry_delay = 1.0 + attempt
                if deadline is not None:
                    remaining_seconds = deadline - time.monotonic()
                    if remaining_seconds <= 0:
                        raise TimeoutError(
                            "Langfuse lookup deadline exceeded"
                        ) from exc
                    retry_delay = min(retry_delay, remaining_seconds)
                if retry_delay > 0:
                    time.sleep(retry_delay)
                continue
            raise
        except urllib.error.HTTPError as exc:
            last_error = exc
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("unexpected langfuse query failure")


def _parse_langfuse_timestamp(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _recent_langfuse_traces(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    name: str | None,
    user_id: str | None,
    start_time: dt.datetime,
    limit: int = 50,
) -> list[dict[str, Any]]:
    params = {
        "limit": str(limit),
        "fields": "core",
        "orderBy": "timestamp.desc",
        "fromTimestamp": start_time.replace(microsecond=0).isoformat(),
    }
    if name:
        params["name"] = name
    if user_id:
        params["userId"] = user_id
    url = f"{query_url.rstrip('/')}/api/public/traces?{urllib.parse.urlencode(params)}"
    payload = _http_get_json(url, public_key, secret_key)
    traces = payload.get("data", [])
    recent: list[dict[str, Any]] = []
    floor = start_time - dt.timedelta(seconds=5)
    for trace in traces:
        timestamp = _parse_langfuse_timestamp(
            trace.get("timestamp") or trace.get("createdAt") or trace.get("updatedAt")
        )
        if timestamp is None or timestamp < floor:
            continue
        recent.append(trace)
    return recent


def _recent_langfuse_all_traces(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    user_id: str | None,
    start_time: dt.datetime,
    session_id: str | None = None,
    limit: int = 100,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    params = {
        "limit": str(limit),
        "fields": "core",
        "orderBy": "timestamp.desc",
        "fromTimestamp": start_time.replace(microsecond=0).isoformat(),
    }
    if user_id:
        params["userId"] = user_id
    if session_id:
        params["sessionId"] = session_id
    url = f"{query_url.rstrip('/')}/api/public/traces?{urllib.parse.urlencode(params)}"
    payload = _http_get_json(
        url,
        public_key,
        secret_key,
        deadline=deadline,
    )
    traces = payload.get("data", [])
    recent: list[dict[str, Any]] = []
    floor = start_time - dt.timedelta(seconds=5)
    for trace in traces:
        timestamp = _parse_langfuse_timestamp(
            trace.get("timestamp") or trace.get("createdAt") or trace.get("updatedAt")
        )
        if timestamp is None or timestamp < floor:
            continue
        recent.append(trace)
    return recent


def _poll_langfuse_session_traces(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    user_id: str | None,
    start_time: dt.datetime,
    session_id: str,
    timeout_seconds: int = 45,
    interval_seconds: float = 3.0,
) -> tuple[list[dict[str, Any]], str | None]:
    deadline = time.time() + timeout_seconds
    traces: list[dict[str, Any]] = []
    last_error: str | None = None
    while True:
        try:
            traces = _recent_langfuse_all_traces(
                query_url=query_url,
                public_key=public_key,
                secret_key=secret_key,
                user_id=user_id,
                start_time=start_time,
                session_id=session_id,
                limit=100,
            )
            last_error = None
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            ConnectionResetError,
            TimeoutError,
        ) as exc:
            traces = []
            last_error = str(exc)
        if traces:
            return traces, last_error
        if time.time() >= deadline:
            return traces, last_error
        time.sleep(interval_seconds)


def _recent_langfuse_generation_observations_for_trace_ids(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    trace_ids: list[str],
    start_time: dt.datetime,
    limit_per_trace: int = 10,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    floor = start_time - dt.timedelta(seconds=5)
    observations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for trace_id in trace_ids:
        url = f"{query_url.rstrip('/')}/api/public/traces/{urllib.parse.quote(trace_id, safe='')}"
        payload = _http_get_json(
            url,
            public_key,
            secret_key,
            deadline=deadline,
        )
        trace_observations = payload.get("observations", [])
        if not isinstance(trace_observations, list):
            continue
        generation_observations = [
            observation
            for observation in trace_observations
            if isinstance(observation, dict) and observation.get("type") == "GENERATION"
        ]
        generation_observations.sort(
            key=lambda observation: _parse_langfuse_timestamp(
                observation.get("startTime")
                or observation.get("createdAt")
                or observation.get("updatedAt")
            )
            or dt.datetime.min.replace(tzinfo=dt.timezone.utc),
            reverse=True,
        )
        for observation in generation_observations[:limit_per_trace]:
            observation_id = observation.get("id")
            if isinstance(observation_id, str) and observation_id in seen_ids:
                continue
            timestamp = _parse_langfuse_timestamp(
                observation.get("startTime")
                or observation.get("createdAt")
                or observation.get("updatedAt")
            )
            if timestamp is None or timestamp < floor:
                continue
            if isinstance(observation_id, str):
                seen_ids.add(observation_id)
            observations.append(observation)
    return observations


def _recent_langfuse_span_observations_for_trace_ids(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    trace_ids: list[str],
    start_time: dt.datetime,
    limit_per_trace: int = 25,
) -> list[dict[str, Any]]:
    floor = start_time - dt.timedelta(seconds=5)
    observations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for trace_id in trace_ids:
        params = {
            "traceId": trace_id,
            "type": "SPAN",
            "limit": str(limit_per_trace),
            "orderBy": "startTime.desc",
            "fields": "core",
        }
        url = f"{query_url.rstrip('/')}/api/public/observations?{urllib.parse.urlencode(params)}"
        payload = _http_get_json(url, public_key, secret_key)
        for observation in payload.get("data", []):
            observation_id = observation.get("id")
            if isinstance(observation_id, str) and observation_id in seen_ids:
                continue
            timestamp = _parse_langfuse_timestamp(
                observation.get("startTime")
                or observation.get("createdAt")
                or observation.get("updatedAt")
            )
            if timestamp is None or timestamp < floor:
                continue
            if isinstance(observation_id, str):
                seen_ids.add(observation_id)
            observations.append(observation)
    return observations


def _extract_generation_metric(
    observation: dict[str, Any], *path: str
) -> Any:
    current: Any = observation
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _route_family_candidates_for_request_route(route: str) -> set[str]:
    normalized = route.split("?", 1)[0].strip().strip("/")
    if not normalized:
        return set()

    route_lower = normalized.lower()
    candidates: set[str] = set()
    parts = [part for part in re.split(r"[/:\-]+", route_lower) if part]
    parts_without_version = [
        part for part in parts if not re.fullmatch(r"v\d+(?:internal)?", part)
    ]
    if parts_without_version:
        candidates.add("_".join(parts_without_version))

    if route_lower in {"anthropic/v1/messages", "anthropic/messages", "v1/messages"}:
        candidates.add("anthropic_messages")
    if route_lower.startswith("openai_passthrough/") and route_lower.endswith(
        "responses"
    ):
        candidates.add("openai_passthrough_responses")

    return candidates


def _is_count_tokens_request_route(request_route: Any) -> bool:
    if not isinstance(request_route, str):
        return False
    normalized = request_route.strip().lower().rstrip("/")
    return normalized.endswith("/count_tokens")


def _generation_observation_matches_allowed_route(
    observation: dict[str, Any], allowed_request_routes: list[str]
) -> bool:
    metadata = observation.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    observed_request_routes: list[str] = []
    request_route = metadata.get("user_api_key_request_route")
    if isinstance(request_route, str):
        observed_request_routes.append(request_route)

    for audit_key in (
        "codex_auto_agent_audit_events",
        "anthropic_auto_agent_audit_events",
        "aawm_alias_routing_audit_events",
    ):
        audit_events = metadata.get(audit_key)
        if not isinstance(audit_events, list):
            continue
        for event in audit_events:
            if not isinstance(event, dict):
                continue
            incoming_endpoint = event.get("incoming_endpoint")
            if isinstance(incoming_endpoint, str):
                observed_request_routes.append(incoming_endpoint)

    if any(
        _is_count_tokens_request_route(route) for route in observed_request_routes
    ):
        return False
    if any(route in allowed_request_routes for route in observed_request_routes):
        return True

    tags = metadata.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tag_values = {tag for tag in tags if isinstance(tag, str)}

    passthrough_route_family = metadata.get("passthrough_route_family")
    for route in allowed_request_routes:
        for route_family in _route_family_candidates_for_request_route(route):
            if passthrough_route_family == route_family:
                return True
            if f"route:{route_family}" in tag_values:
                return True
    return False


def _validate_generation_observations(  # noqa: PLR0915
    *,
    family: str,
    query_url: str,
    public_key: str,
    secret_key: str,
    trace_ids: list[str],
    start_time: dt.datetime,
    allowed_request_routes: list[str] | None = None,
    skip_quality_checks: bool = False,
    allow_zero_cost: bool = False,
    allow_reference_cost_when_invoice_unknown: bool = False,
    allow_unknown_cost_when_invoice_unknown: bool = False,
    preloaded_observations: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    if not trace_ids:
        return [], [], [f"{family} missing trace ids for generation validation"]

    allowed_request_routes = allowed_request_routes or []
    observations: list[dict[str, Any]] = list(preloaded_observations or [])
    route_filtered_observations: list[dict[str, Any]] = []
    last_error: Exception | None = None

    if preloaded_observations is not None:
        route_filtered_observations = observations
        if allowed_request_routes:
            route_filtered_observations = [
                observation
                for observation in observations
                if _generation_observation_matches_allowed_route(
                    observation,
                    allowed_request_routes,
                )
            ]
    else:
        deadline = time.time() + 45
        while True:
            try:
                observations = (
                    _recent_langfuse_generation_observations_for_trace_ids(
                        query_url=query_url,
                        public_key=public_key,
                        secret_key=secret_key,
                        trace_ids=trace_ids,
                        start_time=start_time,
                    )
                )
                last_error = None
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                http.client.RemoteDisconnected,
                ConnectionResetError,
                TimeoutError,
            ) as exc:
                observations = []
                route_filtered_observations = []
                last_error = exc
            else:
                route_filtered_observations = observations
                if allowed_request_routes:
                    route_filtered_observations = [
                        observation
                        for observation in observations
                        if _generation_observation_matches_allowed_route(
                            observation,
                            allowed_request_routes,
                        )
                    ]
                if route_filtered_observations:
                    break

            if time.time() >= deadline:
                break
            time.sleep(3.0)

    if last_error is not None and not observations:
        return [], [], [f"{family} generation lookup failed: {last_error}"]
    if not observations:
        return [], [], [f"{family} missing generation observations"]
    if allowed_request_routes and not route_filtered_observations:
        return [], [], [
            f"{family} missing generation observations for routes: {', '.join(allowed_request_routes)}"
        ]

    summaries: list[dict[str, Any]] = []
    for observation in route_filtered_observations:
        model = observation.get("model") or observation.get("providedModelName")
        prompt_tokens = observation.get("promptTokens")
        if prompt_tokens is None:
            prompt_tokens = _extract_generation_metric(observation, "usageDetails", "input")
        completion_tokens = observation.get("completionTokens")
        if completion_tokens is None:
            completion_tokens = _extract_generation_metric(observation, "usageDetails", "output")
        total_tokens = observation.get("totalTokens")
        if total_tokens is None:
            total_tokens = _extract_generation_metric(observation, "usageDetails", "total")
        cost_total = _extract_generation_metric(observation, "costDetails", "total")
        if cost_total is None:
            cost_total = observation.get("totalCost")
        calculated_total_cost = observation.get("calculatedTotalCost")
        if calculated_total_cost is None:
            calculated_total_cost = observation.get("totalCost")
        metadata = observation.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        actual_invoice_cost_known = metadata.get("actual_invoice_cost_known")
        reference_cost_total = metadata.get("reference_cost_total_usd")
        has_explicit_reference_cost = (
            allow_reference_cost_when_invoice_unknown
            and actual_invoice_cost_known is False
            and isinstance(reference_cost_total, (int, float))
            and not isinstance(reference_cost_total, bool)
            and reference_cost_total > 0
        )
        has_allowed_unknown_invoice_cost = (
            allow_unknown_cost_when_invoice_unknown
            and actual_invoice_cost_known is False
            and cost_total is None
        )
        summary = {
            "id": observation.get("id"),
            "traceId": observation.get("traceId"),
            "name": observation.get("name"),
            "model": model,
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": total_tokens,
            "costDetails.total": cost_total,
            "calculatedTotalCost": calculated_total_cost,
            "actualInvoiceCostKnown": actual_invoice_cost_known,
            "referenceCost.total": reference_cost_total,
            "referenceCostAccepted": has_explicit_reference_cost,
        }
        summaries.append(summary)

        if skip_quality_checks:
            continue

        if not isinstance(model, str) or not model.strip():
            failures.append(f"{family} generation missing model")
        elif model.strip().lower() == "unknown":
            failures.append(f"{family} generation model resolved to unknown")

        if not isinstance(prompt_tokens, (int, float)) or prompt_tokens <= 0:
            failures.append(f"{family} generation missing promptTokens")
        if not isinstance(completion_tokens, (int, float)) or completion_tokens <= 0:
            failures.append(f"{family} generation missing completionTokens")
        if not isinstance(total_tokens, (int, float)) or total_tokens <= 0:
            failures.append(f"{family} generation missing totalTokens")
        if has_explicit_reference_cost or has_allowed_unknown_invoice_cost:
            continue
        if allow_zero_cost:
            if not isinstance(cost_total, (int, float)) or cost_total < 0:
                failures.append(f"{family} generation missing costDetails.total")
            if (
                not isinstance(calculated_total_cost, (int, float))
                or calculated_total_cost < 0
            ):
                failures.append(f"{family} generation missing calculatedTotalCost")
        else:
            if not isinstance(cost_total, (int, float)) or cost_total <= 0:
                failures.append(f"{family} generation missing costDetails.total")
            if (
                not isinstance(calculated_total_cost, (int, float))
                or calculated_total_cost <= 0
            ):
                failures.append(f"{family} generation missing calculatedTotalCost")

    return route_filtered_observations, summaries, sorted(set(failures))


def _validate_span_observations(
    *,
    family: str,
    query_url: str,
    public_key: str,
    secret_key: str,
    trace_ids: list[str],
    start_time: dt.datetime,
    required_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    required_names = required_names or []
    if not required_names:
        return [], [], []
    if not trace_ids:
        return [], [], [f"{family} missing trace ids for span validation"]

    try:
        observations = _recent_langfuse_span_observations_for_trace_ids(
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            trace_ids=trace_ids,
            start_time=start_time,
        )
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        http.client.RemoteDisconnected,
        ConnectionResetError,
        TimeoutError,
    ) as exc:
        return [], [], [f"{family} span lookup failed: {exc}"]

    if not observations:
        return [], [], [f"{family} missing span observations"]

    observed_names = sorted(
        {
            str(observation.get("name")).strip()
            for observation in observations
            if isinstance(observation.get("name"), str) and observation.get("name").strip()
        }
    )
    failures: list[str] = []
    for name in required_names:
        if name not in observed_names:
            failures.append(f"{family} missing span observation: {name}")

    summaries = [
        {
            "id": observation.get("id"),
            "traceId": observation.get("traceId"),
            "name": observation.get("name"),
            "startTime": observation.get("startTime"),
            "endTime": observation.get("endTime"),
        }
        for observation in observations
    ]
    return observations, summaries, sorted(set(failures))


def _collect_trace_tags(traces: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for trace in traces:
        tags = trace.get("tags") or []
        if not isinstance(tags, list):
            continue
        for tag in tags:
            if isinstance(tag, str) and tag not in seen:
                seen.add(tag)
                ordered.append(tag)
    return ordered


def _collect_trace_metadata(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata_items: list[dict[str, Any]] = []
    for trace in traces:
        metadata = trace.get("metadata") or {}
        if isinstance(metadata, dict):
            metadata_items.append(metadata)
    return metadata_items


def _parse_stdout_json_objects(stdout: str) -> list[dict[str, Any]]:
    stripped = stdout.strip()
    if not stripped:
        return []

    objects: list[dict[str, Any]] = []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]

    for line in stripped.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed_line = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed_line, dict):
            objects.append(parsed_line)

    return objects


def _extract_command_session_id(stdout: str) -> str | None:
    for obj in _parse_stdout_json_objects(stdout):
        for key in ("session_id", "sessionId"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_command_thread_id(stdout: str) -> str | None:
    """Extract thread_id from a top-level ``thread.started`` JSONL event.

    Only accepts events whose ``type`` is exactly ``thread.started`` and
    that carry a top-level ``thread_id`` or ``threadId`` string.  This
    prevents conflation with unrelated nested objects or session IDs.
    """
    for obj in _parse_stdout_json_objects(stdout):
        if obj.get("type") != "thread.started":
            continue
        for key in ("thread_id", "threadId"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_trace_environment(trace: dict[str, Any]) -> str | None:
    value = trace.get("environment")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_trace_session_id(trace: dict[str, Any]) -> str | None:
    for key in ("sessionId", "session_id"):
        value = trace.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _validate_trace_context(
    *,
    family: str,
    traces: list[dict[str, Any]],
    expected_environment: str | None = None,
    require_trace_session_id: bool = False,
    expected_trace_session_id: str | None = None,
    require_trace_ids_distinct_from_session_ids: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    trace_ids = sorted(
        {str(trace.get("id")) for trace in traces if isinstance(trace.get("id"), str)}
    )
    environments = sorted(
        {
            environment
            for trace in traces
            if (environment := _extract_trace_environment(trace)) is not None
        }
    )
    session_ids = sorted(
        {
            session_id
            for trace in traces
            if (session_id := _extract_trace_session_id(trace)) is not None
        }
    )
    missing_environment_trace_ids = [
        str(trace.get("id"))
        for trace in traces
        if _extract_trace_environment(trace) is None
    ]
    missing_session_id_trace_ids = [
        str(trace.get("id"))
        for trace in traces
        if _extract_trace_session_id(trace) is None
    ]

    if expected_environment is not None:
        unexpected_environment_trace_ids = [
            str(trace.get("id"))
            for trace in traces
            if _extract_trace_environment(trace) != expected_environment
        ]
        if unexpected_environment_trace_ids:
            failures.append(
                f"{family} trace environment mismatch: expected `{expected_environment}`"
            )
    else:
        unexpected_environment_trace_ids = []

    if require_trace_session_id and missing_session_id_trace_ids:
        failures.append(f"{family} missing trace sessionId")

    if expected_trace_session_id is not None:
        mismatched_session_trace_ids = [
            str(trace.get("id"))
            for trace in traces
            if _extract_trace_session_id(trace) != expected_trace_session_id
        ]
        if mismatched_session_trace_ids:
            failures.append(
                f"{family} trace sessionId mismatch: expected `{expected_trace_session_id}`"
            )
    else:
        mismatched_session_trace_ids = []

    overlapping_trace_session_ids = sorted(set(trace_ids).intersection(session_ids))
    if require_trace_ids_distinct_from_session_ids and overlapping_trace_session_ids:
        failures.append(f"{family} trace ids collapsed into session ids")

    summary = {
        "trace_ids": trace_ids,
        "trace_environments": environments,
        "trace_session_ids": session_ids,
        "missing_environment_trace_ids": missing_environment_trace_ids,
        "missing_session_id_trace_ids": missing_session_id_trace_ids,
        "unexpected_environment_trace_ids": unexpected_environment_trace_ids,
        "mismatched_session_trace_ids": mismatched_session_trace_ids,
        "overlapping_trace_session_ids": overlapping_trace_session_ids,
    }
    return summary, failures


def _validate_trace_enrichment(
    *,
    family: str,
    traces: list[dict[str, Any]],
    required_tags: list[str] | None = None,
    required_tag_prefixes: list[str] | None = None,
    warning_tag_prefixes: list[str] | None = None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    all_tags = _collect_trace_tags(traces)
    metadata_items = _collect_trace_metadata(traces)
    metadata_keys = sorted({key for metadata in metadata_items for key in metadata.keys()})

    for tag in required_tags or []:
        if tag not in all_tags:
            failures.append(f"{family} missing trace tag: {tag}")

    for prefix in required_tag_prefixes or []:
        if not any(tag.startswith(prefix) for tag in all_tags):
            failures.append(f"{family} missing trace tag prefix: {prefix}")

    for prefix in warning_tag_prefixes or []:
        if not any(tag.startswith(prefix) for tag in all_tags):
            warnings.append(f"{family} missing warning trace tag prefix: {prefix}")

    summary = {
        "trace_tags": all_tags,
        "trace_metadata_keys": metadata_keys,
        "warning_tag_prefixes_checked": sorted(warning_tag_prefixes or []),
    }
    return summary, failures, warnings


def _validate_generation_metadata(
    *,
    family: str,
    observations: list[dict[str, Any]],
    required_metadata_truthy: list[str] | None = None,
    required_metadata_minimums: dict[str, int | float] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    metadata_items = [
        observation.get("metadata")
        for observation in observations
        if isinstance(observation.get("metadata"), dict)
    ]
    metadata_items = [metadata for metadata in metadata_items if isinstance(metadata, dict)]
    metadata_keys = sorted({key for metadata in metadata_items for key in metadata.keys()})

    truthy_hits: dict[str, bool] = {}
    for key in required_metadata_truthy or []:
        truthy_hit = any(bool(metadata.get(key)) for metadata in metadata_items)
        truthy_hits[key] = truthy_hit
        if not truthy_hit:
            failures.append(f"{family} missing truthy generation metadata: {key}")

    minimum_hits: dict[str, Any] = {}
    for key, minimum in (required_metadata_minimums or {}).items():
        hit = None
        for metadata in metadata_items:
            value = metadata.get(key)
            if isinstance(value, (int, float)) and value >= minimum:
                hit = value
                break
        minimum_hits[key] = hit
        if hit is None:
            failures.append(f"{family} missing generation metadata >= {minimum}: {key}")

    summary = {
        "generation_metadata_keys": metadata_keys,
        "generation_metadata_truthy_hits": truthy_hits,
        "generation_metadata_minimum_hits": minimum_hits,
    }
    return summary, failures


def _extract_logged_request_body(observation: dict[str, Any]) -> dict[str, Any] | None:
    input_payload = observation.get("input")
    if not isinstance(input_payload, dict):
        return None
    messages = input_payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    first_message = messages[0]
    if not isinstance(first_message, dict):
        return None
    content = first_message.get("content")
    if not isinstance(content, str):
        synthetic_request_body: dict[str, Any] = {}
        if isinstance(observation.get("model"), str):
            synthetic_request_body["model"] = observation["model"]
        if isinstance(messages, list) and messages:
            synthetic_request_body["messages"] = messages
        model_parameters = observation.get("modelParameters")
        if isinstance(model_parameters, dict):
            for key in ("max_tokens", "stream", "stream_options", "reasoning_effort"):
                if key in model_parameters:
                    synthetic_request_body[key] = model_parameters[key]
        return synthetic_request_body or None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_request_body_path_value(
    request_body: dict[str, Any], path: str
) -> Any | None:
    current: Any = request_body
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current.get(segment)
    return current


def _preview_request_body_path_value(value: Any) -> str:
    preview = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(preview) > 160:
        return preview[:157] + "..."
    return preview


def _extract_claude_agent_name_from_observation(observation: dict[str, Any]) -> str | None:
    request_body = _extract_logged_request_body(observation)
    if not isinstance(request_body, dict):
        return None
    match = _CLAUDE_AGENT_NAME_RE.search(json.dumps(request_body))
    if not match:
        return None
    agent_name = match.group(1).strip()
    return agent_name or None


def _collect_text_fragments(value: Any) -> list[str]:
    fragments: list[str] = []
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            fragments.append(value["text"])
        for child in value.values():
            fragments.extend(_collect_text_fragments(child))
    elif isinstance(value, list):
        for child in value:
            fragments.extend(_collect_text_fragments(child))
    elif isinstance(value, str):
        fragments.append(value)
    return fragments


def _validate_logged_request_text_checks(
    *,
    family: str,
    observations: list[dict[str, Any]],
    required_substrings: list[str] | None = None,
    forbidden_substrings: list[str] | None = None,
    warning_required_substrings: list[str] | None = None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    required_substrings = required_substrings or []
    forbidden_substrings = forbidden_substrings or []
    warning_required_substrings = warning_required_substrings or []
    matched_observation_id: str | None = None
    matched_required: dict[str, bool] = {value: False for value in required_substrings}
    matched_warning_required: dict[str, bool] = {
        value: False for value in warning_required_substrings
    }
    forbidden_hits: dict[str, list[str]] = {value: [] for value in forbidden_substrings}

    for observation in observations:
        request_body = _extract_logged_request_body(observation)
        if request_body is None:
            continue
        request_text = "\n".join(_collect_text_fragments(request_body))
        if not request_text:
            continue

        for value in required_substrings:
            if value in request_text:
                matched_required[value] = True

        for value in warning_required_substrings:
            if value in request_text:
                matched_warning_required[value] = True

        current_forbidden_hits = [
            value for value in forbidden_substrings if value in request_text
        ]
        for value in current_forbidden_hits:
            forbidden_hits[value].append(str(observation.get("id")))

        if all(value in request_text for value in required_substrings) and not current_forbidden_hits:
            matched_observation_id = str(observation.get("id"))

    for value, matched in matched_required.items():
        if not matched:
            failures.append(f"{family} missing request substring: {value}")
    for value, matched in matched_warning_required.items():
        if not matched:
            warnings.append(f"{family} missing warning request substring: {value}")
    for value, observation_ids in forbidden_hits.items():
        if observation_ids:
            failures.append(
                f"{family} request still contains forbidden substring `{value}` in {len(observation_ids)} observation(s)"
            )
    if required_substrings and matched_observation_id is None and not failures:
        failures.append(f"{family} missing request observation satisfying text checks")

    summary = {
        "matched_observation_id": matched_observation_id,
        "required_substrings_found": matched_required,
        "warning_required_substrings_found": matched_warning_required,
        "forbidden_substring_hits": forbidden_hits,
    }
    return summary, failures, warnings


def _validate_logged_request_payload_checks(
    *,
    family: str,
    observations: list[dict[str, Any]],
    required_paths: list[str] | None = None,
    warning_present_paths: list[str] | None = None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    required_paths = required_paths or []
    warning_present_paths = warning_present_paths or []

    required_path_found: dict[str, bool] = {path: False for path in required_paths}
    required_path_values: dict[str, list[str]] = {path: [] for path in required_paths}
    warning_path_hits: dict[str, list[dict[str, str]]] = {
        path: [] for path in warning_present_paths
    }

    for observation in observations:
        request_body = _extract_logged_request_body(observation)
        if request_body is None:
            continue

        observation_id = str(observation.get("id"))
        for path in required_paths:
            value = _extract_request_body_path_value(request_body, path)
            if value is None:
                continue
            required_path_found[path] = True
            preview = _preview_request_body_path_value(value)
            if preview not in required_path_values[path]:
                required_path_values[path].append(preview)

        for path in warning_present_paths:
            value = _extract_request_body_path_value(request_body, path)
            if value is None:
                continue
            warning_path_hits[path].append(
                {
                    "observation_id": observation_id,
                    "value": _preview_request_body_path_value(value),
                }
            )

    for path, found in required_path_found.items():
        if not found:
            failures.append(f"{family} missing request payload path: {path}")

    for path, hits in warning_path_hits.items():
        if not hits:
            continue
        observed_values = sorted({hit["value"] for hit in hits})
        warnings.append(
            f"{family} request payload includes warning path `{path}` with value(s): "
            + ", ".join(observed_values)
        )

    summary = {
        "required_paths_found": required_path_found,
        "required_path_values": required_path_values,
        "warning_present_path_hits": warning_path_hits,
    }
    return summary, failures, warnings


def _validate_aawm_dynamic_injection(  # noqa: PLR0915 - Bounded procedural validator.
    *,
    family: str,
    observations: list[dict[str, Any]],
    required_proc: str,
    required_context_keys: list[str] | None = None,
    acceptable_statuses: list[str] | None = None,
    warning_statuses: list[str] | None = None,
    no_memory_required_substrings: list[str] | None = None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    required_context_keys = required_context_keys or []
    acceptable_statuses = acceptable_statuses or ["resolved"]
    warning_statuses = warning_statuses or []
    allowed_statuses = set(acceptable_statuses) | set(warning_statuses)
    no_memory_required_substrings = no_memory_required_substrings or []

    checked_observation_ids: list[str] = []
    proc_values: set[str] = set()
    context_key_values: set[str] = set()
    status_values: list[str] = []
    failed_observation_ids: list[str] = []
    empty_observation_ids: list[str] = []
    resolved_observation_ids: list[str] = []
    request_contains_directive_ids: list[str] = []
    request_contains_failure_block_ids: list[str] = []
    request_missing_no_memory_text_ids: list[str] = []

    for observation in observations:
        metadata = observation.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if "aawm_dynamic_injection_count" not in metadata:
            continue

        checked_observation_ids.append(str(observation.get("id")))
        procs = metadata.get("aawm_dynamic_injection_procs")
        if isinstance(procs, list):
            proc_values.update(
                proc for proc in procs if isinstance(proc, str) and proc.strip()
            )

        context_keys = metadata.get("aawm_dynamic_injection_context_keys")
        if isinstance(context_keys, list):
            context_key_values.update(
                key for key in context_keys if isinstance(key, str) and key.strip()
            )

        statuses = metadata.get("aawm_dynamic_injection_statuses")
        normalized_statuses: list[str] = []
        if isinstance(statuses, list):
            normalized_statuses = [
                status for status in statuses if isinstance(status, str) and status.strip()
            ]
            status_values.extend(normalized_statuses)

        request_body = _extract_logged_request_body(observation)
        request_text = "\n".join(_collect_text_fragments(request_body)) if request_body else ""
        observation_id = str(observation.get("id"))

        if "<!-- AAWM" in request_text:
            request_contains_directive_ids.append(observation_id)
        if (
            f'AAWM "{required_proc}" failed for this session.' in request_text
            or "## AAWM Injection Status" in request_text
        ):
            request_contains_failure_block_ids.append(observation_id)

        if "failed" in normalized_statuses:
            failed_observation_ids.append(observation_id)
        if "empty" in normalized_statuses:
            empty_observation_ids.append(observation_id)
            if any(substring not in request_text for substring in no_memory_required_substrings):
                request_missing_no_memory_text_ids.append(observation_id)
        if "resolved" in normalized_statuses:
            resolved_observation_ids.append(observation_id)

    if not checked_observation_ids:
        failures.append(f"{family} missing AAWM dynamic injection metadata")
    if required_proc not in proc_values:
        failures.append(f"{family} missing AAWM proc: {required_proc}")
    missing_context_keys = [
        key for key in required_context_keys if key not in context_key_values
    ]
    if missing_context_keys:
        failures.append(
            f"{family} missing AAWM context keys: {', '.join(sorted(missing_context_keys))}"
        )
    if failed_observation_ids:
        failures.append(f"{family} AAWM dynamic injection failed")
    if request_contains_directive_ids:
        failures.append(f"{family} request still contains AAWM directive")
    if request_contains_failure_block_ids:
        failures.append(f"{family} request contains AAWM failure block")
    if request_missing_no_memory_text_ids:
        failures.append(f"{family} missing no-memory replacement text")

    unexpected_statuses = sorted({status for status in status_values if status not in allowed_statuses})
    if unexpected_statuses:
        failures.append(
            f"{family} unexpected AAWM injection statuses: {', '.join(unexpected_statuses)}"
        )
    if not any(status in allowed_statuses for status in status_values):
        failures.append(
            f"{family} missing acceptable AAWM injection status: {', '.join(sorted(allowed_statuses))}"
        )
    if empty_observation_ids:
        warnings.append(
            f"{family} AAWM memory not populated for {len(empty_observation_ids)} observation(s)"
        )

    summary = {
        "checked_observation_ids": checked_observation_ids,
        "proc_values": sorted(proc_values),
        "context_key_values": sorted(context_key_values),
        "status_values": status_values,
        "resolved_observation_ids": resolved_observation_ids,
        "empty_observation_ids": empty_observation_ids,
        "failed_observation_ids": failed_observation_ids,
        "request_contains_directive_ids": request_contains_directive_ids,
        "request_contains_failure_block_ids": request_contains_failure_block_ids,
        "request_missing_no_memory_text_ids": request_missing_no_memory_text_ids,
        "warnings": warnings,
    }
    return summary, failures, warnings


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_logged_request_source_files(  # noqa: PLR0915 - Bounded procedural validator.
    *,
    family: str,
    observations: list[dict[str, Any]],
    source_paths_key: str,
    source_hashes_key: str | None = None,
    source_bytes_key: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    checked_observation_ids: list[str] = []
    checked_source_paths: list[str] = []
    content_mismatch_paths: list[str] = []
    metadata_hash_mismatch_paths: list[str] = []
    metadata_bytes_mismatch_paths: list[str] = []
    unreadable_source_paths: list[str] = []

    for observation in observations:
        metadata = observation.get("metadata")
        if not isinstance(metadata, dict):
            continue
        source_paths = metadata.get(source_paths_key)
        if not isinstance(source_paths, list) or not any(
            isinstance(path, str) and path.strip() for path in source_paths
        ):
            continue

        request_body = _extract_logged_request_body(observation)
        if request_body is None:
            failures.append(f"{family} missing logged request body for source verification")
            continue
        request_text = "\n".join(_collect_text_fragments(request_body))
        if not request_text:
            failures.append(f"{family} missing logged request text for source verification")
            continue

        checked_observation_ids.append(str(observation.get("id")))
        source_hashes = metadata.get(source_hashes_key) if source_hashes_key else None
        source_bytes = metadata.get(source_bytes_key) if source_bytes_key else None

        for index, source_path_value in enumerate(source_paths):
            if not isinstance(source_path_value, str) or not source_path_value.strip():
                continue
            source_path = pathlib.Path(source_path_value)
            checked_source_paths.append(str(source_path))
            try:
                file_text = source_path.read_text(encoding="utf-8", errors="replace").rstrip(
                    "\n"
                )
            except Exception:
                unreadable_source_paths.append(str(source_path))
                continue

            actual_hash = _sha256_text(file_text)
            actual_bytes = len(file_text.encode("utf-8"))

            if file_text not in request_text:
                content_mismatch_paths.append(str(source_path))

            if isinstance(source_hashes, list) and index < len(source_hashes):
                metadata_hash = source_hashes[index]
                if isinstance(metadata_hash, str) and metadata_hash != actual_hash:
                    metadata_hash_mismatch_paths.append(str(source_path))

            if isinstance(source_bytes, list) and index < len(source_bytes):
                metadata_size = source_bytes[index]
                if isinstance(metadata_size, int) and metadata_size != actual_bytes:
                    metadata_bytes_mismatch_paths.append(str(source_path))

    if not checked_observation_ids:
        failures.append(f"{family} missing source-file metadata for request verification")
    if unreadable_source_paths:
        failures.append(f"{family} unreadable persisted-output source files")
    if content_mismatch_paths:
        failures.append(f"{family} logged request missing full persisted-output file contents")
    if metadata_hash_mismatch_paths:
        failures.append(f"{family} persisted-output content hash metadata mismatch")
    if metadata_bytes_mismatch_paths:
        failures.append(f"{family} persisted-output byte metadata mismatch")

    summary = {
        "checked_observation_ids": checked_observation_ids,
        "checked_source_paths": checked_source_paths,
        "content_mismatch_paths": content_mismatch_paths,
        "metadata_hash_mismatch_paths": metadata_hash_mismatch_paths,
        "metadata_bytes_mismatch_paths": metadata_bytes_mismatch_paths,
        "unreadable_source_paths": unreadable_source_paths,
    }
    return summary, failures


def _extract_claude_thinking_blocks(observation: dict[str, Any]) -> list[dict[str, Any]]:
    output = observation.get("output")
    if not isinstance(output, dict):
        return []
    thinking_blocks = output.get("thinking_blocks")
    if isinstance(thinking_blocks, list):
        return [block for block in thinking_blocks if isinstance(block, dict)]
    provider_specific_fields = output.get("provider_specific_fields")
    if not isinstance(provider_specific_fields, dict):
        return []
    provider_blocks = provider_specific_fields.get("thinking_blocks")
    if isinstance(provider_blocks, list):
        return [block for block in provider_blocks if isinstance(block, dict)]
    return []


def _observation_has_claude_thinking_signature(observation: dict[str, Any]) -> bool:
    for block in _extract_claude_thinking_blocks(observation):
        signature = block.get("signature")
        if isinstance(signature, str) and signature.strip():
            return True
    return False


def _recent_langfuse_required_name_traces(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    names: list[str],
    user_id: str | None,
    start_time: dt.datetime,
    limit: int = 50,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    recent_by_id: dict[str, dict[str, Any]] = {}
    for name in names:
        params = {
            "limit": str(limit),
            "name": name,
            "fields": "core",
            "orderBy": "timestamp.desc",
            "fromTimestamp": start_time.replace(microsecond=0).isoformat(),
        }
        if user_id:
            params["userId"] = user_id
        url = f"{query_url.rstrip('/')}/api/public/traces?{urllib.parse.urlencode(params)}"
        try:
            payload = _http_get_json(
                url,
                public_key,
                secret_key,
                deadline=deadline,
            )
            traces = payload.get("data", [])
        except (urllib.error.HTTPError, urllib.error.URLError, http.client.RemoteDisconnected):
            traces = _recent_langfuse_all_traces(
                query_url=query_url,
                public_key=public_key,
                secret_key=secret_key,
                user_id=user_id,
                start_time=start_time,
                limit=max(limit, 100),
                deadline=deadline,
            )
        for trace in traces:
            trace_name = trace.get("name")
            if trace_name != name:
                continue
            trace_id = trace.get("id")
            if isinstance(trace_id, str):
                recent_by_id[trace_id] = trace
    return list(recent_by_id.values())


def _poll_langfuse_required_name_traces(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    names: list[str],
    user_id: str | None,
    start_time: dt.datetime,
    limit: int = 50,
    timeout_seconds: int = 45,
    interval_seconds: float = 3.0,
) -> tuple[list[dict[str, Any]], str | None]:
    deadline = time.time() + timeout_seconds
    traces: list[dict[str, Any]] = []
    last_error: str | None = None
    while True:
        try:
            traces = _recent_langfuse_required_name_traces(
                query_url=query_url,
                public_key=public_key,
                secret_key=secret_key,
                names=names,
                user_id=user_id,
                start_time=start_time,
                limit=limit,
            )
            last_error = None
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            ConnectionResetError,
            TimeoutError,
        ) as exc:
            traces = []
            last_error = str(exc)
        actual_names = {trace.get("name") for trace in traces if trace.get("name")}
        if all(name in actual_names for name in names):
            return traces, last_error
        if time.time() >= deadline:
            return traces, last_error
        time.sleep(interval_seconds)


def _poll_langfuse_named_traces(
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    names: list[str],
    user_id: str | None,
    start_time: dt.datetime,
    limit: int = 50,
    timeout_seconds: int = 45,
    interval_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    deadline = time.time() + timeout_seconds
    traces: list[dict[str, Any]] = []
    while True:
        traces = []
        try:
            for name in names:
                traces.extend(
                    _recent_langfuse_traces(
                        query_url=query_url,
                        public_key=public_key,
                        secret_key=secret_key,
                        name=name,
                        user_id=user_id,
                        start_time=start_time,
                        limit=limit,
                    )
                )
        except (urllib.error.HTTPError, urllib.error.URLError, http.client.RemoteDisconnected):
            all_recent = _recent_langfuse_all_traces(
                query_url=query_url,
                public_key=public_key,
                secret_key=secret_key,
                user_id=user_id,
                start_time=start_time,
                limit=max(limit, 100),
            )
            traces = [
                trace
                for trace in all_recent
                if trace.get("name") in names
            ]
        unique = {
            trace.get("id"): trace
            for trace in traces
            if isinstance(trace.get("id"), str)
        }
        traces = list(unique.values())
        actual_names = {trace.get("name") for trace in traces if trace.get("name")}
        if all(name in actual_names for name in names):
            return traces
        if time.time() >= deadline:
            return traces
        time.sleep(interval_seconds)


def _decode_partial_output(value: Any) -> str:
    """Decode subprocess partial output that may be str or bytes.

    ``subprocess.TimeoutExpired`` may carry bytes for stdout/stderr even when
    ``text=True`` was requested (CPython behaviour varies by platform/version).
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _run_command(
    command: list[str],
    *,
    extra_env: dict[str, str] | None = None,
    timeout_seconds: int = 300,
    output_max_chars: int | None = None,
) -> dict[str, Any]:
    env = _scrubbed_child_env(extra_env)
    effective_command, settings_overlay_path = _append_claude_settings_overlay(
        command,
        extra_env=extra_env,
    )
    started = time.time()
    try:
        completed = subprocess.run(
            effective_command,
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = round(time.time() - started, 3)
        raw_stdout = _decode_partial_output(exc.stdout)
        raw_stderr = _decode_partial_output(exc.stderr)
        raw_stdout = raw_stdout.strip()
        raw_stderr = raw_stderr.strip()
        max_chars = (
            _cli_output_max_chars()
            if output_max_chars is None
            else max(0, int(output_max_chars))
        )
        stdout, stdout_truncated = _truncate_captured_text(raw_stdout, max_chars)
        stderr, stderr_truncated = _truncate_captured_text(raw_stderr, max_chars)
        return {
            "command": effective_command,
            "command_string": " ".join(shlex.quote(part) for part in effective_command),
            "exit_code": -1,
            "duration_seconds": duration,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "stdout_original_chars": len(raw_stdout),
            "stderr_original_chars": len(raw_stderr),
            "output_max_chars": max_chars,
            "response_excerpt": _response_excerpt(stdout, stderr),
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "phase": "client_subprocess",
        }
    finally:
        if settings_overlay_path is not None:
            try:
                settings_overlay_path.unlink()
            except FileNotFoundError:
                pass
    duration = round(time.time() - started, 3)
    raw_stdout = (completed.stdout or "").strip()
    raw_stderr = (completed.stderr or "").strip()
    max_chars = (
        _cli_output_max_chars() if output_max_chars is None else max(0, int(output_max_chars))
    )
    stdout, stdout_truncated = _truncate_captured_text(raw_stdout, max_chars)
    stderr, stderr_truncated = _truncate_captured_text(raw_stderr, max_chars)
    return {
        "command": effective_command,
        "command_string": " ".join(shlex.quote(part) for part in effective_command),
        "exit_code": completed.returncode,
        "duration_seconds": duration,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "stdout_original_chars": len(raw_stdout),
        "stderr_original_chars": len(raw_stderr),
        "output_max_chars": max_chars,
        "response_excerpt": _response_excerpt(stdout, stderr),
    }


def _response_excerpt(stdout: str, stderr: str, limit: int = 300) -> str:
    text = stdout or stderr
    text = text.replace("\n", " ").strip()
    return text[:limit]


def _is_claude_command(command: list[str]) -> bool:
    if not command:
        return False
    return pathlib.Path(command[0]).name == "claude"


def _claude_settings_overlay_env(extra_env: dict[str, str] | None) -> dict[str, str]:
    if not extra_env:
        return {}
    overlay_keys = ("ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS")
    return {
        key: str(extra_env[key])
        for key in overlay_keys
        if key in extra_env and extra_env[key] is not None
    }


def _append_claude_settings_overlay(
    command: list[str],
    *,
    extra_env: dict[str, str] | None,
) -> tuple[list[str], pathlib.Path | None]:
    if not _is_claude_command(command) or "--settings" in command:
        return command, None

    overlay_env = _claude_settings_overlay_env(extra_env)
    if not overlay_env:
        return command, None

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="litellm-claude-harness-",
        suffix=".settings.json",
        delete=False,
    )
    settings_path = pathlib.Path(handle.name)
    with handle:
        json.dump({"env": overlay_env}, handle, sort_keys=True)
        handle.write("\n")
    return [*command, "--settings", str(settings_path)], settings_path


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _docker_status() -> str:
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=litellm-dev", "--format", "{{.Status}}"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _inject_codex_session_id(command: list[str]) -> tuple[list[str], str]:
    """Append a run-unique Codex session ID header if not already present.

    Uses the existing ``-c model_providers.<profile>.http_headers.session_id=...``
    mechanics.  Returns (possibly-new command, generated session id).
    A preexisting explicit session override is preserved as-is.
    """
    # Find the profile from -p <profile>
    profile = "litellm"
    for i, token in enumerate(command):
        if token == "-p" and i + 1 < len(command):
            profile = command[i + 1]
            break

    # Check for preexisting session_id override
    session_prefix = f"model_providers.{profile}.http_headers.session_id="
    for token in command:
        if session_prefix in token:
            # Extract existing value (strip quotes)
            raw = token.split(session_prefix, 1)[1].strip().strip('"').strip("'")
            if raw:
                return command, raw

    run_session = f"acceptance-{uuid.uuid4().hex[:16]}"
    injected = list(command)
    injected.extend([
        "-c",
        f'model_providers.{profile}.http_headers.session_id="{run_session}"',
    ])
    return injected, run_session


def _validate_codex(
    config: dict[str, Any],
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
) -> dict[str, Any]:
    started = _utcnow()
    effective_command, run_session = _inject_codex_session_id(config["command"])
    run = _run_command(effective_command, timeout_seconds=int(config.get("timeout_seconds", 300)))
    _record_family_run_evidence(run)
    command_session_id = _extract_command_session_id(run["stdout"])
    command_thread_id = _extract_command_thread_id(run["stdout"])
    post_run_wait_seconds = float(config.get("post_run_wait_seconds", 0) or 0)
    if post_run_wait_seconds > 0:
        time.sleep(post_run_wait_seconds)
    expected_trace_names = config.get("expected_trace_names", [])
    expected_user_ids = config.get("expected_user_ids", [])
    traces, lookup_error = _poll_langfuse_session_traces(
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        user_id=expected_user_ids[0] if expected_user_ids else None,
        start_time=started,
        session_id=run_session,
        timeout_seconds=int(config.get("langfuse_poll_timeout_seconds", 45)),
    )
    actual_trace_names = sorted({trace.get("name") for trace in traces if trace.get("name")})
    actual_user_ids = sorted({trace.get("userId") for trace in traces if trace.get("userId")})
    trace_ids = [trace.get("id") for trace in traces if trace.get("id")]
    failures: list[str] = []
    warnings: list[str] = []
    if not traces:
        failures.append(
            f"codex session trace lookup returned no traces for session {run_session}"
        )
    if lookup_error:
        warnings.append(f"Codex Langfuse session lookup warning: {lookup_error}")
    if run.get("timed_out"):
        failures.append(
            f"codex command timed out after {run.get('timeout_seconds', '?')}s"
        )
    elif run["exit_code"] != 0:
        failures.append("codex command failed")
    failures.extend(
        _enforce_minimum_trace_count(family="codex", traces=traces, config=config)
    )
    for name in expected_trace_names:
        if name not in actual_trace_names:
            failures.append(f"missing Codex trace name: {name}")
    for user_id in expected_user_ids:
        if user_id not in actual_user_ids:
            failures.append(f"missing Codex user id: {user_id}")
    if bool(config.get("require_trace_user_id")) and traces and not actual_user_ids:
        failures.append("Codex traces did not include a Langfuse userId")
    skip_quality_checks, allow_zero_cost = _generation_quality_flags(config)
    (
        _raw_generation_observations,
        generation_observations,
        generation_failures,
    ) = _validate_generation_observations(
        family="codex",
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        trace_ids=trace_ids,
        start_time=started,
        allowed_request_routes=config.get("allowed_generation_routes"),
        skip_quality_checks=skip_quality_checks,
        allow_zero_cost=allow_zero_cost,
        allow_reference_cost_when_invoice_unknown=bool(
            config.get("allow_reference_cost_when_invoice_unknown")
        ),
    )
    failures.extend(generation_failures)
    trace_enrichment_summary, trace_enrichment_failures, trace_enrichment_warnings = _validate_trace_enrichment(
        family="codex",
        traces=traces,
        required_tags=config.get("required_trace_tags"),
        required_tag_prefixes=config.get("required_trace_tag_prefixes"),
        warning_tag_prefixes=config.get("warning_trace_tag_prefixes"),
    )
    failures.extend(trace_enrichment_failures)
    warnings.extend(trace_enrichment_warnings)
    trace_context_summary, trace_context_failures = _validate_trace_context(
        family="codex",
        traces=traces,
        expected_environment=config.get("expected_trace_environment"),
        require_trace_session_id=bool(config.get("require_trace_session_id")),
        expected_trace_session_id=config.get("expected_trace_session_id"),
        require_trace_ids_distinct_from_session_ids=bool(
            config.get("require_trace_ids_distinct_from_session_ids")
        ),
    )
    failures.extend(trace_context_failures)
    generation_metadata_summary, generation_metadata_failures = _validate_generation_metadata(
        family="codex",
        observations=_raw_generation_observations,
        required_metadata_truthy=config.get("required_generation_metadata_truthy"),
        required_metadata_minimums=config.get("required_generation_metadata_minimums"),
    )
    failures.extend(generation_metadata_failures)
    _, span_observations, span_failures = _validate_span_observations(
        family="codex",
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        trace_ids=trace_ids,
        start_time=started,
        required_names=config.get("required_span_names"),
    )
    failures.extend(span_failures)
    return {
        **run,
        "streaming_checked": config.get("streaming_checked", False),
        "langfuse": {
            "expected_trace_names": expected_trace_names,
            "actual_trace_names": actual_trace_names,
            "expected_user_ids": expected_user_ids,
            "actual_user_ids": actual_user_ids,
            "trace_ids": trace_ids,
            "trace_count": len(traces),
            "command_session_id": command_session_id,
            "command_thread_id": command_thread_id,
            "run_session": run_session,
            "trace_context": trace_context_summary,
            "trace_enrichment": trace_enrichment_summary,
            "generation_metadata": generation_metadata_summary,
            "span_observations": span_observations,
            "generation_observations": generation_observations,
        },
        "passed": not failures,
        "failures": sorted(set(failures)),
        "warnings": sorted(set(warnings)),
    }


def _validate_claude(  # noqa: PLR0915 - Bounded family acceptance validator.
    config: dict[str, Any],
    *,
    query_url: str,
    public_key: str,
    secret_key: str,
    fanout_mode: str = "minimal",
) -> dict[str, Any]:
    fanout_modes = config.get("fanout_modes", {})
    selected_mode = fanout_modes.get(fanout_mode, {})
    effective_config = dict(config)
    if selected_mode:
        effective_config.update(selected_mode)
    effective_config = _ensure_claude_harness_headers(
        effective_config,
        target=str(effective_config.get("target_profile") or "local"),
        case_name=f"claude_{fanout_mode}",
    )

    started = _utcnow()
    run = _run_command(
        effective_config["command"],
        extra_env=effective_config.get("env"),
        timeout_seconds=int(effective_config.get("timeout_seconds", 300)),
    )
    _record_family_run_evidence(run)
    command_session_id = _extract_command_session_id(run["stdout"])
    post_run_wait_seconds = float(effective_config.get("post_run_wait_seconds", 0) or 0)
    if post_run_wait_seconds > 0:
        time.sleep(post_run_wait_seconds)
    required_trace_names = effective_config.get("required_trace_names", [])
    expected_user_ids = effective_config.get("expected_user_ids", [])
    if isinstance(command_session_id, str) and command_session_id.strip():
        traces, lookup_error = _poll_langfuse_session_traces(
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            user_id=expected_user_ids[0] if expected_user_ids else None,
            start_time=started,
            session_id=command_session_id.strip(),
            timeout_seconds=int(
                effective_config.get("langfuse_poll_timeout_seconds", 60)
            ),
        )
    else:
        traces, lookup_error = _poll_langfuse_required_name_traces(
            query_url=query_url,
            public_key=public_key,
            secret_key=secret_key,
            names=required_trace_names,
            user_id=expected_user_ids[0] if expected_user_ids else None,
            start_time=started,
            limit=100,
            timeout_seconds=int(
                effective_config.get("langfuse_poll_timeout_seconds", 60)
            ),
        )
    actual_trace_names = sorted({trace.get("name") for trace in traces if trace.get("name")})
    actual_user_ids = sorted({trace.get("userId") for trace in traces if trace.get("userId")})
    trace_ids = [trace.get("id") for trace in traces if trace.get("id")]
    failures: list[str] = []
    warnings: list[str] = []
    if run.get("timed_out"):
        failures.append(
            f"claude command timed out after {run.get('timeout_seconds', '?')}s"
        )
    elif run["exit_code"] != 0:
        failures.append("claude command failed")
    failures.extend(
        _enforce_minimum_trace_count(
            family="claude",
            traces=traces,
            config=effective_config,
        )
    )
    if lookup_error:
        warnings.append(f"Claude Langfuse lookup warning: {lookup_error}")
    for user_id in expected_user_ids:
        if user_id not in actual_user_ids:
            failures.append(f"missing Claude user id: {user_id}")
    if bool(effective_config.get("require_trace_user_id")) and traces and not actual_user_ids:
        failures.append("Claude traces did not include a Langfuse userId")
    skip_quality_checks, allow_zero_cost = _generation_quality_flags(effective_config)
    (
        raw_generation_observations,
        generation_observations,
        generation_failures,
    ) = _validate_generation_observations(
        family="claude",
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        trace_ids=trace_ids,
        start_time=started,
        allowed_request_routes=effective_config.get("allowed_generation_routes"),
        skip_quality_checks=skip_quality_checks,
        allow_zero_cost=allow_zero_cost,
        allow_reference_cost_when_invoice_unknown=bool(
            effective_config.get("allow_reference_cost_when_invoice_unknown")
        ),
    )
    failures.extend(generation_failures)
    observed_agents = sorted(
        {
            agent_name
            for observation in raw_generation_observations
            if (agent_name := _extract_claude_agent_name_from_observation(observation))
        }
    )
    required_agent_names = sorted(
        {
            name.removeprefix("claude-code.")
            for name in required_trace_names
            if isinstance(name, str) and name.strip()
        }
    )
    for agent_name in required_agent_names:
        if agent_name not in observed_agents:
            failures.append(f"missing Claude agent observation: {agent_name}")
    if "orchestrator" not in observed_agents:
        failures.append("missing Claude orchestrator observation")
    if len([name for name in observed_agents if name != "orchestrator"]) == 0:
        failures.append("missing Claude persona/subagent observations")
    filtered_trace_ids = sorted(
        {
            observation.get("traceId")
            for observation in raw_generation_observations
            if isinstance(observation.get("traceId"), str)
        }
    )
    filtered_traces = [
        trace for trace in traces if trace.get("id") in set(filtered_trace_ids)
    ]
    trace_enrichment_summary, trace_enrichment_failures, trace_enrichment_warnings = _validate_trace_enrichment(
        family="claude",
        traces=filtered_traces,
        required_tags=effective_config.get("required_trace_tags"),
        required_tag_prefixes=effective_config.get("required_trace_tag_prefixes"),
        warning_tag_prefixes=effective_config.get("warning_trace_tag_prefixes"),
    )
    failures.extend(trace_enrichment_failures)
    warnings.extend(trace_enrichment_warnings)
    trace_context_summary, trace_context_failures = _validate_trace_context(
        family="claude",
        traces=filtered_traces,
        expected_environment=effective_config.get("expected_trace_environment"),
        require_trace_session_id=bool(effective_config.get("require_trace_session_id")),
        expected_trace_session_id=(
            command_session_id
            if effective_config.get("match_trace_session_id_from_stdout")
            else effective_config.get("expected_trace_session_id")
        ),
        require_trace_ids_distinct_from_session_ids=bool(
            effective_config.get("require_trace_ids_distinct_from_session_ids")
        ),
    )
    failures.extend(trace_context_failures)
    generation_metadata_summary, generation_metadata_failures = _validate_generation_metadata(
        family="claude",
        observations=raw_generation_observations,
        required_metadata_truthy=effective_config.get("required_generation_metadata_truthy"),
        required_metadata_minimums=effective_config.get("required_generation_metadata_minimums"),
    )
    failures.extend(generation_metadata_failures)
    request_text_checks = effective_config.get("request_text_checks", {})
    request_text_summary, request_text_failures, request_text_warnings = _validate_logged_request_text_checks(
        family="claude",
        observations=raw_generation_observations,
        required_substrings=request_text_checks.get("required_substrings"),
        forbidden_substrings=request_text_checks.get("forbidden_substrings"),
        warning_required_substrings=request_text_checks.get(
            "warning_required_substrings"
        ),
    )
    failures.extend(request_text_failures)
    warnings.extend(request_text_warnings)
    request_payload_checks = effective_config.get("request_payload_checks", {})
    request_payload_summary, request_payload_failures, request_payload_warnings = (
        _validate_logged_request_payload_checks(
            family="claude",
            observations=raw_generation_observations,
            required_paths=request_payload_checks.get("required_paths"),
            warning_present_paths=request_payload_checks.get(
                "warning_present_paths"
            ),
        )
    )
    failures.extend(request_payload_failures)
    warnings.extend(request_payload_warnings)
    aawm_dynamic_injection_config = effective_config.get("aawm_dynamic_injection", {})
    aawm_dynamic_injection_summary, aawm_dynamic_injection_failures, aawm_dynamic_injection_warnings = (
        _validate_aawm_dynamic_injection(
            family="claude",
            observations=raw_generation_observations,
            required_proc=aawm_dynamic_injection_config.get(
                "required_proc", "get_agent_memories"
            ),
            required_context_keys=aawm_dynamic_injection_config.get(
                "required_context_keys"
            ),
            acceptable_statuses=aawm_dynamic_injection_config.get(
                "acceptable_statuses"
            ),
            warning_statuses=aawm_dynamic_injection_config.get("warning_statuses"),
            no_memory_required_substrings=aawm_dynamic_injection_config.get(
                "no_memory_required_substrings"
            ),
        )
    )
    failures.extend(aawm_dynamic_injection_failures)
    warnings.extend(aawm_dynamic_injection_warnings)
    source_file_verification_config = effective_config.get(
        "request_source_file_verification", {}
    )
    source_file_summary, source_file_failures = _validate_logged_request_source_files(
        family="claude",
        observations=raw_generation_observations,
        source_paths_key=source_file_verification_config.get(
            "source_paths_key", "claude_persisted_output_source_paths"
        ),
        source_hashes_key=source_file_verification_config.get(
            "source_hashes_key", "claude_persisted_output_source_content_hashes"
        ),
        source_bytes_key=source_file_verification_config.get(
            "source_bytes_key", "claude_persisted_output_source_bytes"
        ),
    )
    failures.extend(source_file_failures)
    _, span_observations, span_failures = _validate_span_observations(
        family="claude",
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        trace_ids=filtered_trace_ids,
        start_time=started,
        required_names=effective_config.get("required_span_names"),
    )
    failures.extend(span_failures)
    claude_signature_observed = any(
        _observation_has_claude_thinking_signature(observation)
        for observation in raw_generation_observations
    )
    return {
        **run,
        "streaming_checked": effective_config.get("streaming_checked", False),
        "langfuse": {
            "fanout_mode": fanout_mode,
            "required_trace_names": required_trace_names,
            "actual_trace_names": actual_trace_names,
            "expected_user_ids": expected_user_ids,
            "actual_user_ids": actual_user_ids,
            "trace_ids": trace_ids,
            "trace_count": len(traces),
            "lookup_error": lookup_error,
            "filtered_trace_ids": filtered_trace_ids,
            "command_session_id": command_session_id,
            "observed_agents": observed_agents,
            "required_agent_names": required_agent_names,
            "trace_context": trace_context_summary,
            "trace_enrichment": trace_enrichment_summary,
            "generation_metadata": generation_metadata_summary,
            "request_text_checks": request_text_summary,
            "request_payload_checks": request_payload_summary,
            "aawm_dynamic_injection": aawm_dynamic_injection_summary,
            "request_source_file_verification": source_file_summary,
            "span_observations": span_observations,
            "thought_signature_observed": claude_signature_observed,
            "generation_observations": generation_observations,
        },
        "passed": not failures,
        "failures": sorted(set(failures)),
        "warnings": sorted(set(warnings)),
    }


def _build_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    for family, result in results.items():
        for failure in result.get("failures", []):
            failures.append(f"{family}: {failure}")
        for warning in result.get("warnings", []):
            warnings.append(f"{family}: {warning}")
    return {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
    }


def _family_error_result(
    name: str,
    exc: Exception,
    *,
    run_evidence: dict[str, Any] | None = None,
    phase: str = "initialization",
) -> dict[str, Any]:
    """Build a family error result, preserving subprocess evidence if available.

    When ``run_evidence`` is provided (the subprocess completed or timed out
    before the observability phase failed), the artifact retains command,
    exit code, bounded stdout/stderr, and the phase where failure occurred.
    """
    if run_evidence:
        result = dict(run_evidence)
        result["phase"] = phase
        result["streaming_checked"] = False
        result["langfuse"] = {
            "expected_trace_names": [],
            "actual_trace_names": [],
            "expected_user_ids": [],
            "actual_user_ids": [],
            "trace_ids": [],
            "trace_count": 0,
        }
        result["passed"] = False
        result["failures"] = [f"{name} {phase} error: {exc}"]
        result["warnings"] = []
        return result
    return {
        "command": [],
        "command_string": "",
        "exit_code": 1,
        "duration_seconds": 0,
        "stdout": "",
        "stderr": "",
        "response_excerpt": "",
        "phase": phase,
        "streaming_checked": False,
        "langfuse": {
            "expected_trace_names": [],
            "actual_trace_names": [],
            "expected_user_ids": [],
            "actual_user_ids": [],
            "trace_ids": [],
            "trace_count": 0,
        },
        "passed": False,
        "failures": [f"{name} {phase} error: {exc}"],
        "warnings": [],
    }


def _run_family_with_evidence(
    name: str,
    validate_fn: Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a family validator, preserving subprocess evidence on failure.

    If the validator raises after the subprocess has run, the returned result
    retains command, exit code, bounded stdout/stderr, and phase.
    """
    invocation: dict[str, Any] = {}
    token = _CURRENT_FAMILY_INVOCATION.set(invocation)
    try:
        return validate_fn(*args, **kwargs)
    except Exception as exc:
        run_evidence = invocation.get("run_evidence")
        phase = "observability_validation" if run_evidence else "initialization"
        return _family_error_result(name, exc, run_evidence=run_evidence, phase=phase)
    finally:
        _CURRENT_FAMILY_INVOCATION.reset(token)


_CURRENT_FAMILY_INVOCATION: ContextVar[dict[str, Any] | None] = ContextVar(
    "acceptance_family_invocation",
    default=None,
)


def _record_family_run_evidence(run: dict[str, Any]) -> None:
    invocation = _CURRENT_FAMILY_INVOCATION.get()
    if invocation is not None:
        invocation["run_evidence"] = run


# ---------------------------------------------------------------------------
# CFG-003: Alias-config inventory, coverage gate, and transactional refresh
# ---------------------------------------------------------------------------

_AAWM_ALIAS_CONFIG_DIR = ROOT / "litellm" / "proxy" / "aawm_alias_config"
_AAWM_ALIAS_CONFIG_REFRESH_PATH = "/aawm/alias-config/refresh"
_HEALTH_READINESS_PATH = "/health/readiness"

# Providers excluded from eligibility for the transactional swap test.
_TRANSACTIONAL_SWAP_EXCLUDED_PROVIDERS = frozenset(
    {"anthropic", "xai"}
)

# Fields that must never appear in persisted artifacts.
# Substring-matched keys (any key containing these is redacted).
_ARTIFACT_REDACTED_SUBSTRINGS = frozenset(
    {
        "authorization",
        "api_key",
        "secret_key",
        "password",
        "credential",
        "raw_yaml",
        "tool_arguments",
        "tool_argument",
        "tool_input",
        "tool_call_arg",
        "tool_summary",
        "raw_prompt",
        "prompt_text",
        "prompt_body",
        "raw_output",
        "_preview",
    }
)
# Exact-matched keys (only these exact key names are redacted).
_ARTIFACT_REDACTED_EXACT = frozenset(
    {
        "yaml",
        "log_excerpt",
        "_log_text",
        "prompt",
        "command",
        "command_string",
        "stdout",
        "stderr",
        "token",
        "arguments",
        "raw_prompt",
        "prompt_text",
        "raw_text",
        "traceback_text",
        "raw_output",
        "tool_input",
        "tool_output",
        "tool_result",
        "input_preview",
        "output_preview",
    }
)

# Defect 3: principled content-key classifier for tool_result/tool_output fields.
_TOOL_CONTENT_PREFIXES = ("tool_result", "tool_output")
_TOOL_CONTENT_SEMANTICS = frozenset(
    {
        "text", "body", "preview", "raw", "output", "result",
        "content", "data", "payload", "message", "response", "value",
    }
)
_TOOL_STRUCTURED_SEMANTICS = frozenset(
    {
        "is_error", "count", "status", "line", "number", "bool",
        "passed", "failed", "success", "code", "type", "name", "id",
    }
)


def _is_tool_content_key(key: str) -> bool:
    """Classify whether a key is a tool content field that should be redacted.

    Returns True if the key begins with tool_result or tool_output AND contains
    content semantics but NOT structured semantics.
    """
    key_lower = key.lower()
    if not any(key_lower.startswith(prefix) for prefix in _TOOL_CONTENT_PREFIXES):
        return False
    if any(sem in key_lower for sem in _TOOL_STRUCTURED_SEMANTICS):
        return False
    return any(sem in key_lower for sem in _TOOL_CONTENT_SEMANTICS)


# TUI executables that qualify a case as a real functional case.
_REAL_TUI_EXECUTABLES = frozenset({"codex", "claude"})

# Route rollup statuses that indicate a candidate is NOT currently available.
_UNAVAILABLE_ROUTE_STATUSES = frozenset(
    {"Cooling Down", "Failed", "Exhausted"}
)


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """POST JSON to *url*, returning (status_code, parsed_response_body).

    Never raises on HTTP errors; returns the status and parsed body.
    """
    data = json.dumps(payload).encode("utf-8")
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)
    request = urllib.request.Request(
        url, data=data, headers=merged_headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        parsed = {"raw_excerpt": body[:500]}
    return status, parsed


def _http_get_json_plain(url: str, *, timeout: float = 20.0) -> tuple[int, dict[str, Any]]:
    """GET *url* returning (status_code, parsed_json). Never raises on HTTP errors."""
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
    except (urllib.error.URLError, OSError) as exc:
        return 0, {"error": str(exc)}
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        parsed = {"raw_excerpt": body[:500]}
    return status, parsed


def _extract_refresh_response_hash(response: dict[str, Any]) -> str:
    """Extract the active config hash from a refresh response.

    Handles both top-level (200 success) and nested detail (400 error) shapes.
    """
    if "active_config_hash" in response:
        return str(response["active_config_hash"])
    detail = response.get("detail")
    if isinstance(detail, dict) and "active_config_hash" in detail:
        return str(detail["active_config_hash"])
    return ""


def _extract_refresh_response_version(response: dict[str, Any]) -> str:
    """Extract config_version from a refresh response (top-level or nested)."""
    if "config_version" in response:
        return str(response["config_version"])
    detail = response.get("detail")
    if isinstance(detail, dict) and "config_version" in detail:
        return str(detail["config_version"])
    return ""


def _load_checked_in_alias_config_yaml(
    alias_name: str = "basic",
) -> tuple[str, str]:
    """Load the exact checked-in YAML bytes for *alias_name*.

    Returns (raw_yaml_text, sha256_hex_of_raw_bytes).
    The returned hash is the SOURCE hash (raw bytes), NOT the semantic
    config_hash used by the runtime.
    """
    config_path = _AAWM_ALIAS_CONFIG_DIR / f"{alias_name}.yaml"
    raw_bytes = config_path.read_bytes()
    return raw_bytes.decode("utf-8"), hashlib.sha256(raw_bytes).hexdigest()


def _load_authoritative_startup_config() -> dict[str, Any]:
    """Load the complete startup alias config using the CFG-002 authoritative
    directory discovery/validation/merge/compile APIs.

    Uses ``config_startup.compile_directory`` which performs O_NOFOLLOW
    descriptor-anchored scanning, fail-closed validation (symlinks, duplicates,
    default conflicts, invalid files), and deterministic merge.

    Returns a dict with:
      - snapshot: the compiled RoutingSnapshot (semantic hash/version/aliases)
      - merged_yaml: YAML text suitable for POST /aawm/alias-config/refresh
      - per_file_hashes: {filename: sha256_of_raw_bytes}
      - file_names: sorted list of source file names
      - config_hash: semantic config_hash from the snapshot
      - config_version: semantic config_version from the snapshot
      - aliases: sorted alias names from the snapshot

    Raises on any CFG-002 validation failure (fail-closed).
    """
    import yaml as _yaml

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        compile_directory,
        _scan_inventory,
    )

    config_dir = _AAWM_ALIAS_CONFIG_DIR
    snapshot = compile_directory(config_dir)

    # Capture per-file raw bytes hashes for restoration proof using the
    # AUTHORITATIVE recursive CFG-002 discovery path (_scan_inventory), not a
    # top-level iterdir.  This sees nested files too, so a nested/multi-file
    # source set is detected and can be failed closed by callers.
    per_file_hashes: dict[str, str] = {}
    inventory = _scan_inventory(config_dir)
    # inventory.files already excludes __init__.py and non-YAML, and captures
    # raw bytes + content_digest deterministically (recursive).
    yaml_files: list[tuple[str, bytes]] = []
    for inv_file in inventory.files:
        per_file_hashes[inv_file.relative_name] = inv_file.content_digest
        yaml_files.append((inv_file.relative_name, inv_file.raw_bytes))
    yaml_files.sort(key=lambda item: item[0])

    # Produce merged YAML for refresh posting.  This is a reserialized merge,
    # NOT raw-YAML-equal to any single source file.
    all_aliases: list[Any] = []
    merged_defaults: dict[str, Any] | None = None
    for _rel_name, raw_bytes in yaml_files:
        doc = _yaml.safe_load(raw_bytes.decode("utf-8"))
        if not isinstance(doc, dict):
            continue
        defaults = doc.get("defaults")
        if isinstance(defaults, dict) and defaults:
            if merged_defaults is None:
                merged_defaults = defaults
        aliases = doc.get("aliases")
        if isinstance(aliases, list):
            all_aliases.extend(aliases)

    merged: dict[str, Any] = {}
    if merged_defaults is not None:
        merged["defaults"] = merged_defaults
    merged["aliases"] = all_aliases
    merged_yaml = _yaml.dump(
        merged, default_flow_style=False, sort_keys=True, allow_unicode=False
    )

    return {
        "snapshot": snapshot,
        "merged_yaml": merged_yaml,
        "per_file_hashes": per_file_hashes,
        "file_names": sorted(per_file_hashes.keys()),
        "config_hash": snapshot.config_hash,
        "config_version": snapshot.config_version,
        "aliases": sorted(snapshot.aliases.keys()),
    }


def _recursive_yaml_source_inventory(
    config_dir: pathlib.Path | None = None,
) -> dict[str, str]:
    """Return the authoritative recursive YAML source inventory.

    Uses the CFG-002 ``_scan_inventory`` discovery path (O_NOFOLLOW,
    descriptor-anchored, recursive) so nested files are included.  Returns
    {relative_name: content_digest} for every accepted YAML source file.
    """
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        _scan_inventory,
    )

    base = config_dir or _AAWM_ALIAS_CONFIG_DIR
    inventory = _scan_inventory(base)
    return {inv_file.relative_name: inv_file.content_digest for inv_file in inventory.files}


# Maximum age for fresh availability evidence (used as default parameter).
_AVAILABILITY_FRESHNESS_SECONDS = 3600.0


def _derive_ingresses_from_snapshot(
    snapshot: Any, alias_name: str
) -> list[str]:
    """Derive supported ingresses for *alias_name* from a compiled
    RoutingSnapshot, using compiler-projected route families including
    anthropic_route_family.
    """
    concrete_candidates = _concretize_alias_candidates(snapshot, alias_name=alias_name)
    ingresses: set[str] = set()
    for cand in concrete_candidates:
        rf = cand.route_family or ""
        if rf.startswith("codex_") or rf == "codex_responses":
            ingresses.add("codex_responses")
        if rf.startswith("anthropic_"):
            ingresses.add("anthropic_messages")
        arf = cand.anthropic_route_family or ""
        if arf:
            ingresses.add("anthropic_messages")
    return sorted(ingresses)


def _concretize_alias_candidates(
    snapshot: Any,
    alias_name: str,
    *,
    path: tuple[str, ...] = (),
) -> list[Any]:
    """Resolve alias references to concrete candidates.

    Mirrors snapshot compiler semantics for reference expansion by following
    ``AliasReference`` entries in priority order and skipping already-visited
    aliases to preserve cycle behavior.
    """
    if alias_name in path:
        return []

    alias = snapshot.aliases.get(alias_name)
    if alias is None:
        return []

    next_path = (*path, alias_name)

    # Dispatch-only aliases should inherit their resolved target's concrete
    # candidate set for inventory/derivation purposes, matching runtime
    # snapshot selection behavior.
    if alias.dispatch is not None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
            _resolve_dispatch_target,
        )

        resolved_target = _resolve_dispatch_target(
            alias_name=alias_name,
            client_product_label=None,
            snapshot=snapshot,
        )
        if resolved_target is None:
            return []
        return _concretize_alias_candidates(
            snapshot,
            alias_name=resolved_target,
            path=next_path,
        )

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
        AliasReference,
    )
    concrete_candidates: list[Any] = []
    for candidate in getattr(alias, "candidates", ()):  # robust against fake snapshots
        if isinstance(candidate, AliasReference):
            concrete_candidates.extend(
                _concretize_alias_candidates(
                    snapshot,
                    alias_name=candidate.alias_name,
                    path=next_path,
                )
            )
            continue
        concrete_candidates.append(candidate)
    return concrete_candidates


def _derive_eligible_candidates_from_snapshot(
    snapshot: Any,
    *,
    alias_name: str = "basic",
    excluded_providers: frozenset[str] | None = None,
    availability_evidence: dict[str, str] | None = None,
    positive_availability: dict[Any, Any] | None = None,
    require_availability: bool = False,
    environment: str = "dev",
    now: dt.datetime | None = None,
    freshness_seconds: float = _AVAILABILITY_FRESHNESS_SECONDS,
) -> list[dict[str, Any]]:
    """Derive ordered eligible candidates from a compiled RoutingSnapshot.

    Filters by excluded_providers, schedule windows, and (when provided)
    runtime availability evidence.

    When ``require_availability`` is True, a candidate is eligible ONLY if
    ``positive_availability`` contains a boundary-valid record for the EXACT
    ``(provider, model)`` identity (validated by ``_availability_record_is_valid``);
    missing/unknown/stale/future-skewed/wrong-env/wrong-provider/wrong-model
    evidence cannot pass.  This is the transactional-refresh path.

    Otherwise, when ``availability_evidence`` (legacy model->status map) is
    provided, a candidate is excluded only if its model is in
    _UNAVAILABLE_ROUTE_STATUSES.

    Returns list of dicts: provider, model, route_family,
    anthropic_route_family, priority, original_index.
    """
    if excluded_providers is None:
        excluded_providers = _TRANSACTIONAL_SWAP_EXCLUDED_PROVIDERS

    if snapshot.aliases.get(alias_name) is None:
        raise ValueError(f"alias {alias_name!r} not found in snapshot")

    alias_candidates = _concretize_alias_candidates(snapshot, alias_name=alias_name)

    now_utc = now if now is not None else dt.datetime.now(dt.timezone.utc)
    eligible: list[dict[str, Any]] = []
    for idx, cand in enumerate(alias_candidates):
        if cand.provider in excluded_providers:
            continue
        # Schedule window check.
        if cand.schedule is not None:
            if not (cand.schedule.start <= now_utc <= cand.schedule.end):
                continue
        # Positive availability requirement (strict boundary validation).
        if require_availability:
            info = (positive_availability or {}).get(
                _availability_key(cand.provider, cand.model)
            )
            if not _availability_record_is_valid(
                info,
                provider=cand.provider,
                model=cand.model,
                environment=environment,
                now=now_utc,
                freshness_seconds=freshness_seconds,
            ):
                continue
        elif availability_evidence is not None:
            # Legacy negative-filter: exclude only explicit unavailable statuses.
            status = availability_evidence.get(cand.model, "")
            if status in _UNAVAILABLE_ROUTE_STATUSES:
                continue
        eligible.append(
            {
                "provider": cand.provider,
                "model": cand.model,
                "route_family": cand.route_family or "",
                "anthropic_route_family": cand.anthropic_route_family or "",
                "priority": cand.priority,
                "original_index": idx,
            }
        )

    eligible.sort(key=lambda c: c["priority"], reverse=True)
    return eligible


def _derive_eligible_candidates_from_yaml(
    raw_yaml: str,
    *,
    alias_name: str = "basic",
    excluded_providers: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse YAML and derive ordered currently-eligible candidates.

    Legacy helper for tests that do not have a compiled snapshot.
    Returns a list of dicts with keys: provider, model, route_family, priority,
    original_index.
    """
    import yaml as _yaml

    if excluded_providers is None:
        excluded_providers = _TRANSACTIONAL_SWAP_EXCLUDED_PROVIDERS

    doc = _yaml.safe_load(raw_yaml)
    if not isinstance(doc, dict):
        raise ValueError("alias config YAML must be a mapping")
    aliases = doc.get("aliases")
    if not isinstance(aliases, list):
        raise ValueError("alias config YAML must contain an 'aliases' list")

    target_alias: dict[str, Any] | None = None
    for alias in aliases:
        if isinstance(alias, dict) and alias.get("name") == alias_name:
            target_alias = alias
            break
    if target_alias is None:
        raise ValueError(f"alias {alias_name!r} not found in config YAML")

    candidates_raw = target_alias.get("candidates")
    if not isinstance(candidates_raw, list):
        raise ValueError(f"alias {alias_name!r} has no candidates list")

    now = dt.datetime.now(dt.timezone.utc)
    eligible: list[dict[str, Any]] = []
    for idx, cand in enumerate(candidates_raw):
        if not isinstance(cand, dict):
            continue
        provider = str(cand.get("provider", ""))
        if provider in excluded_providers:
            continue
        schedule = cand.get("schedule")
        if isinstance(schedule, dict):
            start_str = schedule.get("start", "")
            end_str = schedule.get("end", "")
            try:
                start = dt.datetime.fromisoformat(str(start_str).replace("Z", "+00:00"))
                end = dt.datetime.fromisoformat(str(end_str).replace("Z", "+00:00"))
                if not (start <= now <= end):
                    continue
            except (ValueError, TypeError):
                continue
        eligible.append(
            {
                "provider": provider,
                "model": str(cand.get("model", "")),
                "route_family": str(cand.get("route_family", "")),
                "anthropic_route_family": str(cand.get("anthropic_route_family", "")),
                "priority": int(cand.get("priority", 0)),
                "original_index": idx,
            }
        )

    eligible.sort(key=lambda c: c["priority"], reverse=True)
    return eligible


def _build_priority_swap_yaml(
    raw_yaml: str,
    *,
    alias_name: str = "basic",
    excluded_providers: frozenset[str] | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Build an in-memory YAML copy swapping only the first two eligible
    candidates' effective priorities.

    Returns (swapped_yaml_text, original_eligible, swapped_eligible).
    Never modifies the source file.
    """
    import yaml as _yaml

    original_eligible = _derive_eligible_candidates_from_yaml(
        raw_yaml, alias_name=alias_name, excluded_providers=excluded_providers
    )
    if len(original_eligible) < 2:
        raise ValueError(
            f"need at least 2 eligible candidates for priority swap, "
            f"got {len(original_eligible)}"
        )

    first = original_eligible[0]
    second = original_eligible[1]

    doc = _yaml.safe_load(raw_yaml)
    aliases = doc.get("aliases", [])
    target_alias: dict[str, Any] | None = None
    for alias in aliases:
        if isinstance(alias, dict) and alias.get("name") == alias_name:
            target_alias = alias
            break
    if target_alias is None:
        raise ValueError(f"alias {alias_name!r} not found")

    candidates_raw = target_alias.get("candidates", [])
    first_idx = first["original_index"]
    second_idx = second["original_index"]
    candidates_raw[first_idx]["priority"] = second["priority"]
    candidates_raw[second_idx]["priority"] = first["priority"]

    swapped_yaml = _yaml.dump(
        doc, default_flow_style=False, sort_keys=False, allow_unicode=False
    )
    swapped_eligible = _derive_eligible_candidates_from_yaml(
        swapped_yaml, alias_name=alias_name, excluded_providers=excluded_providers
    )
    return swapped_yaml, original_eligible, swapped_eligible


def _build_exact_pair_priority_swap_yaml(
    raw_yaml: str,
    *,
    pair: tuple[tuple[str, str], tuple[str, str]],
    alias_name: str = "basic",
    excluded_providers: frozenset[str] | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a YAML copy swapping priorities of an EXACT (provider, model) pair.

    Contract (CFG-003 body B finding c):
    - ``pair`` is ``((provider_a, model_a), (provider_b, model_b))`` -- the two
      availability-evidenced identities to swap.
    - Fails closed (raises ValueError) BEFORE producing YAML when:
      * either identity is absent from the eligible candidate list;
      * either identity is ambiguous (appears more than once);
      * the two identities are not distinct;
      * fewer than 2 eligible candidates exist.
    - The swap targets the EXACT requested pair, never the positional first-two
      raw candidates.  If the first two raw candidates differ from the requested
      pair, only the requested pair is swapped.
    - Returns (swapped_yaml, original_eligible, swapped_eligible).
    - Never modifies the source file.

    Existing callers using ``_build_priority_swap_yaml`` are unaffected.
    """
    import yaml as _yaml

    (provider_a, model_a), (provider_b, model_b) = pair
    if (provider_a, model_a) == (provider_b, model_b):
        raise ValueError(
            f"exact-pair swap requires two distinct identities, got duplicate: "
            f"({provider_a}, {model_a})"
        )

    original_eligible = _derive_eligible_candidates_from_yaml(
        raw_yaml, alias_name=alias_name, excluded_providers=excluded_providers
    )
    if len(original_eligible) < 2:
        raise ValueError(
            f"need at least 2 eligible candidates for exact-pair swap, "
            f"got {len(original_eligible)}"
        )

    # Locate each identity in the eligible list -- must be present exactly once.
    def _find_unique(provider: str, model: str) -> dict[str, Any]:
        matches = [
            c for c in original_eligible
            if c["provider"] == provider and c["model"] == model
        ]
        if len(matches) == 0:
            raise ValueError(
                f"exact-pair identity ({provider}, {model}) not found among "
                f"eligible candidates; refusing to produce swap YAML"
            )
        if len(matches) > 1:
            raise ValueError(
                f"exact-pair identity ({provider}, {model}) is ambiguous "
                f"({len(matches)} matches); refusing to produce swap YAML"
            )
        return matches[0]

    cand_a = _find_unique(provider_a, model_a)
    cand_b = _find_unique(provider_b, model_b)

    doc = _yaml.safe_load(raw_yaml)
    aliases = doc.get("aliases", [])
    target_alias: dict[str, Any] | None = None
    for alias in aliases:
        if isinstance(alias, dict) and alias.get("name") == alias_name:
            target_alias = alias
            break
    if target_alias is None:
        raise ValueError(f"alias {alias_name!r} not found")

    candidates_raw = target_alias.get("candidates", [])
    idx_a = cand_a["original_index"]
    idx_b = cand_b["original_index"]
    candidates_raw[idx_a]["priority"] = cand_b["priority"]
    candidates_raw[idx_b]["priority"] = cand_a["priority"]

    swapped_yaml = _yaml.dump(
        doc, default_flow_style=False, sort_keys=False, allow_unicode=False
    )
    swapped_eligible = _derive_eligible_candidates_from_yaml(
        swapped_yaml, alias_name=alias_name, excluded_providers=excluded_providers
    )
    return swapped_yaml, original_eligible, swapped_eligible


def _parse_route_availability_evidence(
    log_text: str, alias_name: str
) -> dict[str, str]:
    """Parse runtime route rollup log lines for *alias_name* and return
    a map of model -> latest status.

    Route rollup lines have the shape:
      `` - {model}({alias}):{effort-or-none} - Turns: N [{message}] [{status}] -> {route}``

    The mandatory ``:{effort-or-none}`` segment is the request-specific
    provider-bound effort (``reasoning.effort``, then ``reasoning_effort``,
    then ``output_config.effort``; same-request
    ``reasoning_effort_native_value`` only when the final body lacks a field).
    Absent effort and explicit ``none`` both appear as ``:none``. Mixed
    efforts for the same model remain separate rollup buckets; this parser
    still keys availability by model only and keeps the latest status per
    model so existing CFG-003 consumers stay compatible. Effort is never
    inferred from TOML/YAML, alias defaults, capabilities, or model names.

    Only the latest status per model is retained.  Models not mentioned
    in the logs are absent from the result (treated as unknown/available
    by callers that use this as a filter).
    """
    evidence: dict[str, str] = {}
    marker = f"({alias_name})"
    for line in log_text.splitlines():
        if marker not in line:
            continue
        # Extract model: between " - " and "("
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        rest = stripped[2:]
        paren_idx = rest.find("(")
        if paren_idx < 0:
            continue
        model = rest[:paren_idx].strip()
        if not model:
            continue
        # Extract status: last bracketed token before optional " -> "
        arrow_idx = rest.rfind(" -> ")
        segment = rest[:arrow_idx] if arrow_idx > 0 else rest
        # Find all [status] tokens
        import re as _re
        statuses = _re.findall(r"\[([^\]]+)\]", segment)
        if statuses:
            evidence[model] = statuses[-1]
    return evidence


# ---------------------------------------------------------------------------
# Value-level secret sanitization (CFG-003 body B finding b).
#
# Key-based redaction (below) catches sensitive *keys*, but free-text string
# values (JSONL ``message`` fields, malformed ``raw_excerpt`` blobs, legacy
# ``.log`` lines, error tracebacks) can embed credentials inline.  These
# patterns scrub embedded secrets while preserving bounded diagnostic context
# (the surrounding text and the secret *class* label are retained).
# ---------------------------------------------------------------------------
_SENSITIVE_VALUE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Authorization / Bearer header forms: "Authorization: Bearer sk-...",
    # "bearer <token>", '"authorization": "..."' -- redact the credential run.
    (
        re.compile(
            r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=_-]{8,}"
        ),
        r"\1 [REDACTED]",
    ),
    # Scheme-bearing Authorization forms: "Authorization: Basic <base64>",
    # "Authorization: Token <token>" -- redact the credential after the scheme
    # label while retaining the scheme name for diagnostics.
    (
        re.compile(
            r"(?i)\b(authorization)\b[\s\"':=]+(basic|token|digest|hmac)\s+[A-Za-z0-9._~+/=_-]{6,}"
        ),
        r"\1: \2 [REDACTED]",
    ),
    (
        re.compile(
            r"(?i)\b(authorization)\b[\s\"':=]+[A-Za-z0-9._~+/=_-]{12,}"
        ),
        r"\1=[REDACTED]",
    ),
    # sk-style API tokens (OpenAI/Anthropic style): sk-..., sk-ant-..., sk-proj-...
    (
        re.compile(r"\bsk-[A-Za-z0-9_-]{10,}"),
        "sk-[REDACTED]",
    ),
    # Generic long high-entropy token assigned to a credential-class name:
    # api_key=..., password: ..., credential = ..., secret_key: ..., token=...
    (
        re.compile(
            r"(?i)(?:\b|(?<=_))(api[_-]?key|password|passwd|credential|secret[_-]?key|"
            r"access[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key|"
            r"master[_-]?key|salt[_-]?key)\b\s*[=:]\s*[\"']?[^\s\"',}{)\]]{6,}"
        ),
        r"\1=[REDACTED]",
    ),
)


def _sanitize_sensitive_string_value(text: str) -> str:
    """Scrub embedded secrets from a free-text string value.

    Covers Bearer/Authorization forms, sk-style tokens, and
    api_key/password/credential assignments wherever they appear (including
    inside malformed JSONL excerpts, JSONL ``message`` values, and legacy log
    lines).  Surrounding diagnostic text and the secret class label are
    preserved; only the credential run is replaced with ``[REDACTED]``.
    Strings without embedded secrets are returned unchanged.
    """
    result = text
    for pattern, replacement in _SENSITIVE_VALUE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


# Keys whose values must contain ONLY bounded digest records at any nesting
# depth.  Arbitrary strings, nested raw dicts/lists, legacy entries, and
# malformed digest records are replaced with a redaction marker.
_FORBIDDEN_CONTEXT_DIGEST_KEYS = frozenset(
    {
        "matched_forbidden_contexts",
        "ignored_unattributed_forbidden_contexts",
    }
)
_DIGEST_APPROVED_FIELDS = frozenset({"sha256", "char_count", "line_count"})


def _sanitize_forbidden_context_value(value: Any) -> Any:
    """Validate and sanitize a forbidden-context map value.

    A valid value is a dict mapping substring keys to bounded digest records
    with exactly the approved structural fields (sha256: str, char_count: int,
    line_count: int).  Anything else -- raw strings, nested dicts/lists,
    legacy entries, extra fields, wrong types -- is replaced with
    ``"[REDACTED]"`` so prompt/tool/log text can never survive at any depth.
    """
    if not isinstance(value, dict):
        return "[REDACTED]"
    result: dict[str, Any] = {}
    for key, val in value.items():
        if not isinstance(val, dict):
            result[str(key)] = "[REDACTED]"
            continue
        if set(val.keys()) != _DIGEST_APPROVED_FIELDS:
            result[str(key)] = "[REDACTED]"
            continue
        sha = val.get("sha256")
        chars = val.get("char_count")
        lines = val.get("line_count")
        if not isinstance(sha, str) or isinstance(chars, bool) or isinstance(lines, bool):
            result[str(key)] = "[REDACTED]"
            continue
        if not isinstance(chars, int) or not isinstance(lines, int):
            result[str(key)] = "[REDACTED]"
            continue
        result[str(key)] = {"sha256": sha, "char_count": chars, "line_count": lines}
    return result



def _redact_sensitive_artifact_fields(value: Any) -> Any:
    """Recursively redact sensitive keys from an artifact structure.

    Uses substring matching for credential-class keys and exact matching
    for structured fields (command, stdout, stderr, yaml, prompt, token).
    String values are additionally sanitized to redact embedded secrets
    (Bearer tokens, sk-style keys, api_key/password/credential assignments)
    while preserving useful bounded diagnostic context.
    Never persists credentials, authorization headers, raw prompts, tool
    arguments, raw YAML, commands, or workstation identity.
    """
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, val in value.items():
            key_lower = str(key).lower()
            if key_lower in _ARTIFACT_REDACTED_EXACT:
                result[key] = "[REDACTED]"
            elif any(sub in key_lower for sub in _ARTIFACT_REDACTED_SUBSTRINGS):
                result[key] = "[REDACTED]"
            elif key_lower in _FORBIDDEN_CONTEXT_DIGEST_KEYS:
                result[key] = _sanitize_forbidden_context_value(val)
            elif _is_tool_content_key(key_lower):
                result[key] = "[REDACTED]"
            else:
                result[key] = _redact_sensitive_artifact_fields(val)
        return result
    if isinstance(value, list):
        return [_redact_sensitive_artifact_fields(item) for item in value]
    if isinstance(value, str):
        return _sanitize_sensitive_string_value(value)
    return value


def _is_real_tui_case(case_config: dict[str, Any]) -> bool:
    """Return True only if the case has an actual command list whose first
    element is a codex or claude executable.  cli_passthrough alone does NOT
    qualify; HTTP-only or synthetic cases are rejected.
    """
    cmd = case_config.get("command")
    if isinstance(cmd, list) and cmd:
        executable = str(cmd[0]).rsplit("/", 1)[-1]
        if executable in _REAL_TUI_EXECUTABLES:
            return True
    return False


def _is_egress_case(case_config: dict[str, Any]) -> bool:
    """Return True if the case would perform provider egress of any kind.

    Covers real TUI commands (codex/claude), arbitrary command lists,
    http_request cases, and cli_passthrough cases.  Pure env-gate/skip cases
    (only ``required_env`` etc.) are NOT egress and are not blocked by the
    inventory gate.  Finding 3: unhealthy inventory must block ALL selected
    egress cases before execution, not only Codex/Claude commands.
    """
    if not isinstance(case_config, dict):
        return False
    cmd = case_config.get("command")
    if isinstance(cmd, list) and cmd:
        return True
    if case_config.get("http_request"):
        return True
    if case_config.get("cli_passthrough"):
        return True
    return False


def _validate_alias_ingress_coverage(
    *,
    alias_inventory: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    selected_cases: list[str],
) -> tuple[bool, list[str]]:
    """Require exactly one named real-TUI functional case for each
    alias/ingress pair among the selected cases.

    Coverage must be declared in case metadata (verification_alias +
    verification_ingress) and tied to an actual Codex/Claude command
    executable, not inferred from case names.  HTTP-only or synthetic
    cases are rejected.

    Returns (passed, failures).
    """
    failures: list[str] = []

    required_pairs: set[tuple[str, str]] = set()
    for alias_entry in alias_inventory:
        alias_name = str(alias_entry.get("alias", ""))
        for ingress in alias_entry.get("supported_ingresses", []):
            required_pairs.add((alias_name, str(ingress)))

    declared_pairs: dict[tuple[str, str], list[str]] = {}
    for case_name in selected_cases:
        case_config = cases.get(case_name)
        if not isinstance(case_config, dict):
            continue
        v_alias = case_config.get("verification_alias")
        v_ingress = case_config.get("verification_ingress")
        if not v_alias or not v_ingress:
            continue
        if not _is_real_tui_case(case_config):
            failures.append(
                f"case {case_name!r} declares coverage for "
                f"{v_alias}/{v_ingress} but is not a real TUI case "
                f"(requires codex or claude command list)"
            )
            continue
        pair = (str(v_alias), str(v_ingress))
        declared_pairs.setdefault(pair, []).append(case_name)

    for pair in sorted(required_pairs):
        if pair not in declared_pairs:
            failures.append(
                f"missing real-TUI functional case for alias={pair[0]!r} "
                f"ingress={pair[1]!r}"
            )

    for pair, case_names in sorted(declared_pairs.items()):
        if len(case_names) > 1:
            failures.append(
                f"duplicate coverage for alias={pair[0]!r} ingress={pair[1]!r}: "
                f"cases={case_names}"
            )

    return not failures, failures


def _validate_complete_coverage_map(
    *,
    alias_inventory: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Validate that the COMPLETE configured case map (all cases, not just
    selected) covers every active alias/ingress pair with exactly one real
    TUI case.  Used during ordinary runs to ensure the configured map is
    coherent without requiring every case to be selected.

    Returns (passed, failures).
    """
    failures: list[str] = []

    required_pairs: set[tuple[str, str]] = set()
    for alias_entry in alias_inventory:
        alias_name = str(alias_entry.get("alias", ""))
        for ingress in alias_entry.get("supported_ingresses", []):
            required_pairs.add((alias_name, str(ingress)))

    declared_pairs: dict[tuple[str, str], list[str]] = {}
    for case_name, case_config in cases.items():
        if not isinstance(case_config, dict):
            continue
        v_alias = case_config.get("verification_alias")
        v_ingress = case_config.get("verification_ingress")
        if not v_alias or not v_ingress:
            continue
        if not _is_real_tui_case(case_config):
            continue
        pair = (str(v_alias), str(v_ingress))
        declared_pairs.setdefault(pair, []).append(case_name)

    for pair in sorted(required_pairs):
        if pair not in declared_pairs:
            failures.append(
                f"configured case map missing real-TUI case for "
                f"alias={pair[0]!r} ingress={pair[1]!r}"
            )

    for pair, case_names in sorted(declared_pairs.items()):
        if len(case_names) > 1:
            failures.append(
                f"configured case map has duplicate coverage for "
                f"alias={pair[0]!r} ingress={pair[1]!r}: cases={case_names}"
            )

    return not failures, failures


# ---------------------------------------------------------------------------
# CFG-003: Active error-intake baseline/delta collector (finding 1)
# ---------------------------------------------------------------------------

_ANALYSIS_DIR = ROOT / ".analysis"


def _snapshot_source_inventory(
    config_dir: pathlib.Path | None = None,
) -> dict[str, str]:
    """Snapshot the full recursive source inventory of the alias config dir.

    Returns {relative_path: sha256_hex} for every file (recursive).
    Used to prove no checked-in file was mutated during a run.
    """
    base = config_dir or _AAWM_ALIAS_CONFIG_DIR
    inventory: dict[str, str] = {}
    if not base.is_dir():
        return inventory
    for filepath in sorted(base.rglob("*")):
        if filepath.is_file():
            try:
                inventory[str(filepath.relative_to(base))] = hashlib.sha256(
                    filepath.read_bytes()
                ).hexdigest()
            except OSError:
                inventory[str(filepath.relative_to(base))] = "unreadable"
    return inventory


def _discover_error_intake_files(analysis_dir: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Discover all *-error.jsonl and *-error.log files in root and nested
    .analysis directories (recursive)."""
    base = analysis_dir or _ANALYSIS_DIR
    if not base.is_dir():
        return []
    files: list[pathlib.Path] = []
    for pattern in ("*-error.jsonl", "*-error.log"):
        files.extend(base.rglob(pattern))
    return sorted(files)


def _snapshot_error_intake(
    analysis_dir: pathlib.Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Snapshot file identity/size/line count for all error intake files.

    Returns {relative_path: {size, line_count, inode}} for each discovered file.
    """
    base = analysis_dir or _ANALYSIS_DIR
    snapshot: dict[str, dict[str, Any]] = {}
    for filepath in _discover_error_intake_files(base):
        try:
            stat = filepath.stat()
            raw = filepath.read_bytes()
            line_count = raw.count(b"\n")
            snapshot[str(filepath.relative_to(base))] = {
                "size": stat.st_size,
                "line_count": line_count,
                "inode": stat.st_ino,
            }
        except OSError:
            snapshot[str(filepath.relative_to(base))] = {
                "size": 0,
                "line_count": 0,
                "inode": 0,
                "error": "unreadable",
            }
    return snapshot


def _collect_error_intake_delta(  # noqa: PLR0915
    baseline: dict[str, dict[str, Any]],
    *,
    initiation_time: dt.datetime,
    environment: str = "dev",
    container: str = "litellm-dev",
    case_name: str | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
    strict_correlation: bool = False,
    current_snapshot: dict[str, dict[str, Any]] | None = None,
    analysis_dir: pathlib.Path | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compare current error intake files against baseline and collect new
    events attributable to this run.

    Attribution rules:
    - JSONL: observed_at >= initiation_time AND environment matches AND
      (container matches if present) AND (case/session/trace match if present)
    - Legacy .log: appended lines (line_count > baseline) are attributed
      if they contain the environment string.
    - Truncation/rotation (size decreased or inode changed) fails closed.
    - Malformed JSONL lines are recorded but do not fail.

    When ``strict_correlation`` is True, an event qualifies ONLY by an exact
    matching session_id AND trace_id in its context; the sparse temporal/
    environment/container fallback is disabled so events cannot qualify on
    environment/container/time alone.

    When ``current_snapshot`` is provided, it is reused as the authoritative
    current state instead of taking a second snapshot, eliminating the
    two-snapshot race between collection and baseline advancement.

    Returns (attributed_events, failures).
    """
    base = analysis_dir or _ANALYSIS_DIR
    current = (
        current_snapshot
        if current_snapshot is not None
        else _snapshot_error_intake(base)
    )
    failures: list[str] = []
    attributed: list[dict[str, Any]] = []

    for rel_path, cur_info in sorted(current.items()):
        # Fail closed on unreadable intake files.
        if cur_info.get("error") == "unreadable":
            failures.append(
                f"error intake file unreadable (fail closed): {rel_path}"
            )
            continue
        base_info = baseline.get(rel_path)
        if base_info is None:
            # New file appeared -- fail closed.
            failures.append(f"new error intake file appeared: {rel_path}")
            continue

        # Truncation/rotation detection.
        if cur_info.get("inode") != base_info.get("inode"):
            failures.append(f"error intake file rotated (inode changed): {rel_path}")
            continue
        if cur_info["size"] < base_info["size"]:
            failures.append(f"error intake file truncated: {rel_path}")
            continue

        if cur_info["line_count"] <= base_info["line_count"]:
            continue  # No new lines.

        # Read new lines.
        filepath = base / rel_path
        try:
            raw_lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            failures.append(f"cannot read error intake file {rel_path}: {exc}")
            continue

        new_lines = raw_lines[base_info["line_count"]:]

        if rel_path.endswith(".jsonl"):
            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    # Malformed append -- record but do not fail.
                    attributed.append({
                        "file": rel_path,
                        "malformed": True,
                        "excerpt": line[:200],
                    })
                    continue
                if not isinstance(event, dict):
                    continue
                # Attribution.
                observed_at_str = str(event.get("observed_at", ""))
                try:
                    observed_at = dt.datetime.fromisoformat(observed_at_str)
                    if observed_at.tzinfo is None:
                        observed_at = observed_at.replace(tzinfo=dt.timezone.utc)
                except (ValueError, TypeError):
                    continue
                if observed_at < initiation_time:
                    continue
                if str(event.get("environment", "")) != environment:
                    continue
                ctx = event.get("context")
                if not isinstance(ctx, dict):
                    ctx = {}
                # Identity attribution: for each of container/session/trace/case,
                # an EXPLICITLY present differing value rejects the event; a
                # sparse (absent/empty) value falls through to the conservative
                # temporal+environment+target-container fallback.
                ev_container = ctx.get("container")
                if ev_container and str(ev_container) != container:
                    continue
                ev_session = ctx.get("session_id")
                if session_id and ev_session and str(ev_session) != session_id:
                    continue
                ev_trace = ctx.get("trace_id")
                if trace_id and ev_trace and str(ev_trace) != trace_id:
                    continue
                ev_case = ctx.get("case") or ctx.get("case_name")
                if case_name and ev_case and str(ev_case) != case_name:
                    continue
                # Strict correlation: require an exact matching session_id AND
                # trace_id; the sparse fallback above is not sufficient.
                if strict_correlation:
                    if not (
                        session_id
                        and trace_id
                        and str(ev_session or "") == session_id
                        and str(ev_trace or "") == trace_id
                    ):
                        continue
                attributed.append({
                    "file": rel_path,
                    "observed_at": observed_at_str,
                    "environment": event.get("environment"),
                    "level": event.get("level"),
                    "message": str(event.get("message", ""))[:300],
                    "fingerprint": event.get("fingerprint"),
                    "attributed_container": str(ev_container) if ev_container else None,
                    "attributed_session": str(ev_session) if ev_session else None,
                    "attributed_trace": str(ev_trace) if ev_trace else None,
                    "attributed_case": str(ev_case) if ev_case else None,
                    "sparse_fallback": not (ev_container or ev_session or ev_trace or ev_case),
                })
        else:
            # Legacy .log: in strict correlation mode, unstructured lines can
            # never satisfy exact session+trace identity and must NOT be
            # attributed as correlated evidence.  They are skipped entirely so
            # they cannot satisfy or fail a case.  In non-strict mode they
            # remain attributed by environment substring (diagnostic only).
            if not strict_correlation:
                for line in new_lines:
                    if environment in line:
                        attributed.append({
                            "file": rel_path,
                            "legacy_line": line[:300],
                        })

    # Check for files that disappeared.
    for rel_path in baseline:
        if rel_path not in current:
            failures.append(f"error intake file disappeared: {rel_path}")

    return attributed, failures


def _summarize_error_intake_snapshot(
    snapshot: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Sanitized summary of an error-intake snapshot for artifact persistence.

    Records only file identity/size/line-count metadata -- never file content.
    """
    return {
        "file_count": len(snapshot),
        "files": {
            rel: {
                "size": info.get("size"),
                "line_count": info.get("line_count"),
                "inode": info.get("inode"),
            }
            for rel, info in sorted(snapshot.items())
        },
    }


def _delta_error_intake_summary(
    baseline: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Sanitized per-file delta (line-count / size growth) between two
    error-intake snapshots.  Records only metadata deltas -- never content.
    """
    delta: dict[str, dict[str, Any]] = {}
    all_paths = sorted(set(baseline) | set(current))
    for rel in all_paths:
        base_info = baseline.get(rel, {})
        cur_info = current.get(rel, {})
        base_lines = int(base_info.get("line_count") or 0)
        cur_lines = int(cur_info.get("line_count") or 0)
        base_size = int(base_info.get("size") or 0)
        cur_size = int(cur_info.get("size") or 0)
        delta[rel] = {
            "line_count_delta": cur_lines - base_lines,
            "size_delta": cur_size - base_size,
            "status": (
                "added" if rel not in baseline
                else "removed" if rel not in current
                else "unchanged" if (cur_lines == base_lines and cur_size == base_size)
                else "grown"
            ),
        }
    return {
        "file_count": len(delta),
        "total_line_count_delta": sum(d["line_count_delta"] for d in delta.values()),
        "files": delta,
    }


# ---------------------------------------------------------------------------
# CFG-003: Positive availability evidence (finding 2)
# ---------------------------------------------------------------------------

# Statuses that positively prove availability.
_POSITIVE_AVAILABILITY_STATUSES = frozenset(
    {"available", "selected", "healthy", "success", "active"}
)
# Statuses that prove unavailability.
_NEGATIVE_AVAILABILITY_STATUSES = frozenset(
    {"at capacity", "unavailable", "cooling down", "failed", "exhausted", "cooldown"}
)
# Maximum age for fresh availability evidence.
_AVAILABILITY_FRESHNESS_SECONDS = 3600.0


# Required quota windows per exact (provider, model) identity (Finding 1).
# When a candidate appears here, EVERY listed quota_key must be present in the
# fresh result set and positive for the candidate to qualify.  A required
# window that is missing or stale fails the candidate closed.  Providers/models
# absent from this map preserve single-window compatibility (any observed
# window must be positive, but no specific window is mandated).
_REQUIRED_AVAILABILITY_WINDOWS: dict[tuple[str, str], frozenset[str]] = {
    # Alibaba Token Plan emits an account-wide quota shared across its models,
    # written as paired 5h and 7d windows with a tied observed_at.  Both must
    # be fresh and positive.
    ("alibaba_token_plan", "alibaba_token_plan/qwen3.8-max-preview"): frozenset(
        {"alibaba_token_plan_5h:credits", "alibaba_token_plan_7d:credits"}
    ),
    (
        "alibaba_token_plan",
        "alibaba_token_plan/deepseek-v4-flash-0731",
    ): frozenset(
        {"alibaba_token_plan_5h:credits", "alibaba_token_plan_7d:credits"}
    ),
    ("alibaba_token_plan", "alibaba_token_plan/qwen3.6-flash"): frozenset(
        {"alibaba_token_plan_5h:credits", "alibaba_token_plan_7d:credits"}
    ),
}


def _query_positive_availability_evidence(
    *,
    db_settings: dict[str, Any],
    candidates: list[dict[str, Any]],
    environment: str = "dev",
    freshness_seconds: float = _AVAILABILITY_FRESHNESS_SECONDS,
) -> dict[str, Any]:
    """Query rate_limit_observations for positive availability evidence.

    Evidence is keyed by the EXACT ``(provider, model)`` identity (never model
    alone), so two providers sharing a model name are distinguished.

    A provider/model may emit MULTIPLE quota rows sharing the same
    ``observed_at`` (e.g. Alibaba Token Plan writes 5h and 7d windows with a
    tied timestamp).  ``ORDER BY observed_at DESC LIMIT 1`` would arbitrarily
    observe only one window and could miss an exhausted sibling.  This reader
    therefore evaluates the latest fresh evidence for EVERY distinct
    ``quota_key`` relevant to the exact provider/model, and the candidate is
    positive ONLY when all observed windows are fresh, valid, and
    ``remaining_pct > 0``.  Any exhausted/invalid window fails the candidate
    closed.  Providers that emit a single quota row degrade to the prior
    one-window behavior.  Missing/stale/exhausted evidence does NOT count as
    available.  Generic provider reachability is never used and no evidence is
    fabricated.

    The ``rate_limit_observations`` table has no environment column, so the
    selected target database/profile is bound as the environment and recorded
    explicitly on every record (``environment`` + ``environment_binding``).

    Returns {(provider, model): {available, evidence, observed_at,
             environment, environment_binding}}.
    """
    import psycopg

    result: dict[str, Any] = {}
    try:
        conn = psycopg.connect(
            host=db_settings["host"],
            port=db_settings["port"],
            dbname=db_settings["dbname"],
            user=db_settings["user"],
            password=db_settings["password"],
            connect_timeout=10,
            autocommit=True,
            row_factory=psycopg.rows.dict_row,
        )
    except Exception:  # noqa: BLE001
        return result

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=freshness_seconds)
    try:
        with conn.cursor() as cur:
            for cand in candidates:
                provider = cand["provider"]
                model = cand["model"]
                key = (provider, model)
                cur.execute(
                    """
                    SELECT observed_at, provider, model, quota_key,
                           remaining_pct,
                           quota_remaining, source, evidence
                    FROM public.rate_limit_observations
                    WHERE provider = %s AND model = %s
                      AND observed_at >= %s
                    ORDER BY observed_at DESC
                    """,
                    (provider, model, cutoff),
                )
                rows = cur.fetchall()
                result[key] = _aggregate_quota_window_availability(
                    rows,
                    provider=provider,
                    model=model,
                    environment=environment,
                )
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
    return result


def _availability_timestamp_sort_key(value: dt.datetime | None) -> tuple[int, float]:
    """Stable, overflow-safe sort key for an optional aware datetime.

    Missing/unparseable timestamps sort BEFORE any real timestamp so a
    deterministic maximum prefers genuine observed_at values.  Uses an epoch
    timestamp (float) to avoid ``datetime.min`` tz-offset overflow during
    comparison.
    """
    if value is None:
        return (0, 0.0)
    return (1, value.timestamp())


def _remaining_pct_is_positive(value: Any) -> bool:
    """True only for a real numeric ``remaining_pct`` strictly greater than 0.

    Booleans are rejected (``isinstance(True, int)`` is True) so a malformed
    ``True`` cannot masquerade as positive quota.  Fail-closed on any other
    type.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
    )


def _remaining_pct_sort_key(value: Any) -> tuple[int, float]:
    """Stable sort key for ``remaining_pct``: valid numerics sort by value and
    precede invalid/missing values, giving a deterministic minimum."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value))
    return (1, 0.0)


def _aggregate_quota_window_availability(
    rows: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
    environment: str,
) -> dict[str, Any]:
    """Aggregate multi-window quota rows into a single availability verdict.

    Groups rows by ``quota_key`` and, for each window, evaluates ALL rows tied
    at the maximum ``observed_at`` (Finding 2).  Tied-latest handling is
    deterministic and fail-closed: the window is positive only if EVERY
    tied-latest row is valid and ``remaining_pct > 0``; a single exhausted or
    invalid tied row fails the window.  Reversing row order never changes the
    verdict because the maximum is computed by value and ties are required to
    be unanimously positive.

    The candidate is positive ONLY when every observed window is positive AND
    every required window for the exact provider/model is present (Finding 1).
    A required window that is missing or stale (filtered upstream by the
    freshness cutoff) fails the candidate closed.  Providers/models without an
    explicit required-window contract preserve single-window compatibility.

    No evidence is fabricated; missing rows yield ``no_fresh_row``.
    """
    base = {
        "provider": provider,
        "model": model,
        "environment": environment,
        "environment_binding": "target_db_profile",
    }
    if not rows:
        return {
            **base,
            "available": False,
            "evidence": "no_fresh_row",
            "observed_at": None,
        }

    # Group rows by distinct quota_key.
    rows_by_window: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        qk = str(row.get("quota_key") or "")
        rows_by_window.setdefault(qk, []).append(row)

    # Evaluate every window; ALL tied-latest rows must be positive.
    window_verdicts: list[tuple[str, bool, str]] = []
    all_positive = True
    for qk in sorted(rows_by_window):
        window_rows = rows_by_window[qk]
        # Deterministic maximum observed_at by parsed value (string fallback).
        max_obs = max(
            (_parse_observed_at(r.get("observed_at")) for r in window_rows),
            key=_availability_timestamp_sort_key,
        )
        tied_latest = [
            r for r in window_rows
            if _parse_observed_at(r.get("observed_at")) == max_obs
        ]
        # Fail-closed: EVERY tied-latest row must be valid and positive.
        tied_positive = all(
            _remaining_pct_is_positive(r.get("remaining_pct"))
            for r in tied_latest
        )
        # Deterministic evidence: minimum remaining_pct among tied-latest rows
        # surfaces an exhausted sibling regardless of arrival order.
        min_remaining = min(
            (r.get("remaining_pct") for r in tied_latest),
            key=_remaining_pct_sort_key,
        )
        if tied_positive:
            window_verdicts.append((qk, True, f"remaining_pct={min_remaining}"))
        else:
            all_positive = False
            window_verdicts.append((qk, False, f"remaining_pct={min_remaining}"))

    # Finding 1: enforce the required-window contract for the exact
    # provider/model.  A required window absent from the fresh result set
    # (missing or stale) fails the candidate closed.
    required_windows = _REQUIRED_AVAILABILITY_WINDOWS.get((provider, model))
    missing_windows: list[str] = []
    if required_windows:
        present = set(rows_by_window)
        for rq in sorted(required_windows):
            if rq not in present:
                all_positive = False
                missing_windows.append(rq)

    # Use the most recent observed_at across all rows for the record.
    latest_observed_at = max(
        (r.get("observed_at") for r in rows),
        key=lambda v: _availability_timestamp_sort_key(_parse_observed_at(v)),
    )

    if len(window_verdicts) == 1 and not missing_windows:
        # Single-window provider without a forced contract: preserve the prior
        # evidence format exactly.
        _, positive, evidence_str = window_verdicts[0]
        return {
            **base,
            "available": positive,
            "evidence": evidence_str,
            "observed_at": str(latest_observed_at),
        }

    # Multi-window provider: report per-window verdicts deterministically.
    evidence_parts = [f"{qk}={verdict}" for qk, positive, verdict in window_verdicts]
    evidence_parts.extend(f"{rq}=missing" for rq in missing_windows)
    return {
        **base,
        "available": all_positive,
        "evidence": "; ".join(evidence_parts),
        "observed_at": str(latest_observed_at),
    }


def _availability_key(provider: str, model: str) -> tuple[str, str]:
    """Composite identity key for positive availability evidence."""
    return (provider, model)


def _parse_observed_at(value: Any) -> dt.datetime | None:
    """Parse an ``observed_at`` value into a timezone-aware UTC datetime.

    Returns None for missing/malformed values (fail-closed).  Naive
    datetimes are assumed UTC.
    """
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _availability_record_is_valid(
    record: Any,
    *,
    provider: str,
    model: str,
    environment: str,
    now: dt.datetime | None = None,
    freshness_seconds: float = _AVAILABILITY_FRESHNESS_SECONDS,
    clock_skew_tolerance_seconds: float = 60.0,
) -> bool:
    """Validate a positive-availability record at the boundary (defect 1).

    Requires ALL of:
    - record is a dict with ``available is True`` (not merely truthy);
    - the record's provider, model, environment, and environment_binding
      fields are ALL present and match the expected exact values;
    - a parseable ``observed_at`` within ``freshness_seconds`` of ``now``
      (injected or current UTC); future-skewed beyond the clock-skew
      tolerance fails closed;

    Missing ANY required field, or a stale/future-skewed/absent/wrong-env/
    wrong-provider/wrong-model/malformed-timestamp value, fails closed.  This
    does NOT rely only on the SQL cutoff or stamped labels.
    """
    if not isinstance(record, dict):
        return False
    if record.get("available") is not True:
        return False
    # Required identity fields must be present AND match exactly.
    rec_provider = record.get("provider")
    if rec_provider is None or str(rec_provider) != provider:
        return False
    rec_model = record.get("model")
    if rec_model is None or str(rec_model) != model:
        return False
    rec_env = record.get("environment")
    if rec_env is None or str(rec_env) != environment:
        return False
    rec_binding = record.get("environment_binding")
    if rec_binding is None or str(rec_binding) != "target_db_profile":
        return False
    # observed_at must be present and parseable within the freshness window.
    observed_at = _parse_observed_at(record.get("observed_at"))
    if observed_at is None:
        return False
    now_utc = now if now is not None else dt.datetime.now(dt.timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=dt.timezone.utc)
    age = (now_utc - observed_at).total_seconds()
    if age > freshness_seconds:
        return False  # stale
    if age < -clock_skew_tolerance_seconds:
        return False  # future-skewed beyond tolerance
    return True


def _candidate_order_matches(
    observed_order: Any,
    expected_candidates: list[dict[str, Any]],
) -> bool:
    """Compare the EXACT complete ordered candidate identity lists.

    Requires the observed ``active_candidate_order`` list to match the
    expected eligible candidates element-for-element with NO prefix
    acceptance and NO extra tail.  Each element must match provider, model,
    route_family, anthropic_route_family, priority, and last_resort.
    The last_resort key must be present in BOTH observed and expected;
    missing key on either side fails the match.
    """
    if not isinstance(observed_order, list):
        return False
    if len(observed_order) != len(expected_candidates):
        return False
    for obs, exp in zip(observed_order, expected_candidates):
        if not isinstance(obs, dict):
            return False
        if obs.get("provider") != exp["provider"]:
            return False
        if obs.get("model") != exp["model"]:
            return False
        if obs.get("route_family") != exp["route_family"]:
            return False
        if obs.get("anthropic_route_family") != exp.get("anthropic_route_family", ""):
            return False
        if obs.get("priority") != exp["priority"]:
            return False
        # Defect 2: last_resort key must be present in BOTH and values must match.
        if "last_resort" not in obs or "last_resort" not in exp:
            return False
        if bool(obs["last_resort"]) != bool(exp["last_resort"]):
            return False
    return True


def _derive_full_order_from_snapshot(
    snapshot: Any,
    alias_name: str = "basic",
) -> list[dict[str, Any]]:
    """Derive the COMPLETE candidate order directly from the snapshot.

    Independent of provider exclusions, availability, and schedule windows.
    Returns all candidates in their original snapshot order with normalized
    last_resort bool (finding 3).
    """
    alias = snapshot.aliases.get(alias_name)
    if alias is None:
        return []
    return [
        {
            "provider": cand.provider,
            "model": cand.model,
            "route_family": cand.route_family or "",
            "anthropic_route_family": cand.anthropic_route_family or "",
            "priority": cand.priority,
            "last_resort": cand.priority == 0,
        }
        for cand in alias.candidates
    ]


def _serialize_availability_evidence(
    evidence: dict[Any, Any],
) -> list[dict[str, Any]]:
    """Convert tuple-keyed availability evidence into a JSON-safe, sanitized
    list of records preserving the exact (provider, model) identity."""
    records: list[dict[str, Any]] = []
    for key, info in sorted(evidence.items(), key=lambda kv: tuple(map(str, kv[0]))):
        if isinstance(key, tuple) and len(key) == 2:
            provider, model = key
        else:
            provider, model = "", str(key)
        record = {"provider": provider, "model": model}
        if isinstance(info, dict):
            record.update(info)
        records.append(record)
    return records


def _filter_candidates_by_positive_availability(
    candidates: list[dict[str, Any]],
    availability: dict[str, Any],
    *,
    environment: str = "dev",
    now: dt.datetime | None = None,
    freshness_seconds: float = _AVAILABILITY_FRESHNESS_SECONDS,
) -> list[dict[str, Any]]:
    """Filter candidates to only those with a boundary-valid positive
    availability record for the EXACT (provider, model) identity (finding 1).

    A record must have ``available is True``, a parseable ``observed_at``
    within the freshness window relative to ``now``, and matching
    environment/binding.  Missing/unknown/stale/future-skewed/wrong-env/
    malformed evidence cannot pass."""
    return [
        c for c in candidates
        if _availability_record_is_valid(
            availability.get(_availability_key(c["provider"], c["model"])),
            provider=c["provider"],
            model=c["model"],
            environment=environment,
            now=now,
            freshness_seconds=freshness_seconds,
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local CLI acceptance checks through litellm-dev.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to suite config JSON.")
    parser.add_argument("--write-artifact", default=None, help="Where to write the JSON artifact.")
    parser.add_argument("--langfuse-query-url", default=None, help="Override Langfuse query URL.")
    parser.add_argument(
        "--claude-fanout-mode",
        choices=("minimal", "full"),
        default="minimal",
        help="Claude fan-out validation depth.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target profile (dev/prod). Rewrites family routes consistently.",
    )
    parser.add_argument(
        "--resolve-target-json",
        action="store_true",
        help="Resolve and validate the target profile, print JSON, and exit.",
    )
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    config = _load_suite_config(config_path)

    target_profile = _resolve_target_profile(config, target=args.target)
    _apply_target_profile(config, target_profile)
    if args.resolve_target_json:
        print(  # noqa: T201 - intentional machine-readable CLI output
            json.dumps(_public_target_profile(target_profile), sort_keys=True)
        )
        return 0
    if not args.write_artifact:
        parser.error("--write-artifact is required unless --resolve-target-json is used")
    artifact_path = pathlib.Path(args.write_artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    container_env_cache: dict[tuple[str, str], str] = {}
    public_key, secret_key = _resolve_langfuse_credentials(
        config,
        target_profile,
        container_env_cache=container_env_cache,
    )
    query_url = args.langfuse_query_url or os.environ.get("LANGFUSE_QUERY_URL") or config.get(
        "langfuse_query_url", "http://127.0.0.1:3000"
    )

    artifact: dict[str, Any] = {
        "suite_version": config.get("suite_version", 1),
        "timestamp": _isoformat(_utcnow()),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "environment": {
            "litellm_base_url": target_profile["litellm_base_url"],
            "langfuse_query_url": query_url,
            "docker_litellm_dev_status": _docker_status(),
            "target_profile": _public_target_profile(target_profile),
        },
        "results": {},
        "summary": {},
    }

    artifact["results"]["codex"] = _run_family_with_evidence(
        "codex",
        _validate_codex,
        config["codex"],
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
    )

    artifact["results"]["claude"] = _run_family_with_evidence(
        "claude",
        _validate_claude,
        config["claude"],
        query_url=query_url,
        public_key=public_key,
        secret_key=secret_key,
        fanout_mode=args.claude_fanout_mode,
    )

    artifact["summary"] = _build_summary(artifact["results"])
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(  # noqa: T201 - CLI emits the final machine-readable summary.
        json.dumps(artifact["summary"], indent=2)
    )
    return 0 if artifact["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
