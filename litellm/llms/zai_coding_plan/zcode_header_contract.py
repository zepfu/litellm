"""Load the published ZCode header descriptor and build unsigned model headers.

This module reads the local ZCode contract and assembles identity headers for
Z.AI Coding Plan model requests. It does not sign, perform HTTP, or use SSH.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

ZCODE_HEADER_CONTRACT_PATH_ENV = "LITELLM_ZCODE_NATIVE_HEADER_CONTRACT_PATH"
ZCODE_DESCRIPTOR_RELATIVE_PATH = (
    ".analysis/zai/20260922/zcode-native-header-contract.json"
)

_EXPECTED_SCHEMA_VERSION = 2
_EXPECTED_CLIENT = "zcode"
_EXPECTED_ENDPOINT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
# Source handshake timeout. The descriptor does not publish this value.
_HANDSHAKE_TIMEOUT_SECONDS = 10.0
_PROFILE_NAMES = frozenset({"app_server", "cli"})
_ALLOWED_SESSION_TYPES = frozenset({"main", "subagent", "other"})
_SESSION_ID_FIELDS = (
    "session_id",
    "aawm_session_id",
    "codex_session_id",
    "claude_session_id",
    "anthropic_session_id",
    "prompt_cache_key",
)
_SESSION_PREFIXES = ("sess_", "subagent_agent_")
_QUERY_PREFIX = "query_"
_EXACT_BLOCKED_HEADER_NAMES = frozenset(
    {
        "cookie",
        "x-api-key",
        "x-device-mid",
        "x-msh-device-id",
        "x-opencode-session",
    }
)
_BLOCKED_HEADER_PREFIXES = ("x-cursor-", "x-opencode-")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SDK_OPENAI_COMPATIBLE_TOKEN_RE = re.compile(
    r"ai-sdk/openai-compatible/\d+\.\d+\.\d+(?!\d)"
)
_SDK_PROVIDER_UTILS_TOKEN_RE = re.compile(r"ai-sdk/provider-utils/\d+\.\d+\.\d+(?!\d)")
_CacheKey = tuple[str, int, int]

_CONTRACT_CACHE: dict[_CacheKey, "ZCodeHeaderContract"] = {}


class ZCodeHeaderContractError(Exception):
    """Raised when the ZCode header contract cannot be loaded or applied."""


@dataclass(frozen=True)
class ZCodeSigningMaterial:
    """Unsigned signing endpoints and timeouts from the ZCode descriptor."""

    runtime_header_version: str
    feature_gate_url: str
    feature_gate_timeout_seconds: float
    feature_gate_cache_ttl_seconds: float
    handshake_url: str
    handshake_timeout_seconds: float


@dataclass(frozen=True)
class ZCodeHeaderContract:
    """Published ZCode identity headers and signing material for model requests."""

    schema_version: int
    runtime_header_version: str
    appimage_sha256: str
    profiles: Mapping[str, Mapping[str, str]]
    sdk_openai_compatible_token: str
    sdk_provider_utils_token: str
    forbidden_inbound_headers: frozenset[str]
    signing: ZCodeSigningMaterial
    source_path: str


def load_zcode_header_contract(path: Optional[str] = None) -> ZCodeHeaderContract:
    """Load and cache the published ZCode header descriptor."""

    resolved = _resolve_contract_path(path)
    try:
        stat_result = resolved.stat()
    except OSError:
        raise ZCodeHeaderContractError(
            "ZCode header contract could not be read"
        ) from None
    cache_key = (str(resolved), stat_result.st_mtime_ns, stat_result.st_size)
    cached = _CONTRACT_CACHE.get(cache_key)
    if cached is None:
        cached = _parse_contract(resolved, _load_json_object(resolved))
        _CONTRACT_CACHE[cache_key] = cached
    return _detach_contract(cached)


def build_zcode_model_headers(
    contract: ZCodeHeaderContract,
    *,
    api_key: str,
    litellm_params: Optional[Mapping[str, Any]],
    profile_name: Optional[str] = None,
    runtime_globals: Optional[Mapping[str, Any]] = None,
    request_id_factory: Optional[Callable[[], str]] = None,
    optional_params: Optional[Mapping[str, Any]] = None,
) -> dict[str, str]:
    """Build unsigned ZCode model-request headers from a loaded contract."""

    if not isinstance(contract, ZCodeHeaderContract):
        raise ZCodeHeaderContractError("contract must be a ZCodeHeaderContract")
    if not isinstance(api_key, str) or api_key.strip() == "":
        raise ZCodeHeaderContractError("api_key must be a non-blank string")

    sources = _session_sources(litellm_params, optional_params)
    headers = zcode_profile_headers(
        contract, litellm_params=litellm_params, profile_name=profile_name
    )
    profile_user_agent = headers.get("User-Agent")
    if not isinstance(profile_user_agent, str) or profile_user_agent.strip() == "":
        raise ZCodeHeaderContractError("selected ZCode profile is missing User-Agent")

    assembled = {
        **headers,
        "User-Agent": _assemble_user_agent(
            profile_user_agent, contract, runtime_globals
        ),
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        **_attribution_headers(sources, request_id_factory),
    }
    _reject_forbidden_headers(assembled, contract.forbidden_inbound_headers)
    return assembled


def zcode_profile_headers(
    contract: ZCodeHeaderContract,
    *,
    litellm_params: Optional[Mapping[str, Any]] = None,
    profile_name: Optional[str] = None,
) -> dict[str, str]:
    """Return a copy of the selected ZCode profile headers."""

    if not isinstance(contract, ZCodeHeaderContract):
        raise ZCodeHeaderContractError("contract must be a ZCodeHeaderContract")
    sources = _metadata_dicts(litellm_params)
    selected_profile = _select_profile_name(profile_name, sources)
    profile_headers = contract.profiles.get(selected_profile)
    if not isinstance(profile_headers, Mapping):
        raise ZCodeHeaderContractError("selected ZCode header profile is missing")
    return _copy_string_mapping(profile_headers)


def reset_zcode_header_contract_cache() -> None:
    """Drop cached descriptor parses so the next load reads the file again."""

    _CONTRACT_CACHE.clear()


def _resolve_contract_path(path: Optional[str]) -> Path:
    """Resolve an explicit path, then the env override, then cwd and file parents."""

    if path is not None:
        if not isinstance(path, str):
            raise ZCodeHeaderContractError(
                "ZCode header contract path must be a string"
            )
        if path.strip() != "":
            return _resolve_existing_file(
                path, "ZCode header contract path does not exist"
            )

    env_path = os.environ.get(ZCODE_HEADER_CONTRACT_PATH_ENV)
    if env_path is not None and env_path.strip() != "":
        return _resolve_existing_file(
            env_path,
            f"{ZCODE_HEADER_CONTRACT_PATH_ENV} does not point to an existing ZCode header contract file",
        )

    relative_path = Path(ZCODE_DESCRIPTOR_RELATIVE_PATH)
    for directory in _search_directories():
        candidate = directory / relative_path
        if candidate.is_file():
            return candidate.resolve()
    raise ZCodeHeaderContractError(
        "ZCode header contract was not found at "
        f"{ZCODE_DESCRIPTOR_RELATIVE_PATH}. Set {ZCODE_HEADER_CONTRACT_PATH_ENV}."
    )


def _resolve_existing_file(path: str, message: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except OSError:
        raise ZCodeHeaderContractError(message) from None
    if not resolved.is_file():
        raise ZCodeHeaderContractError(message)
    return resolved


def _search_directories() -> list[Path]:
    directories: list[Path] = []
    seen: set[Path] = set()

    def add_chain(start: Path) -> None:
        try:
            chain = (start, *tuple(start.parents))
        except OSError:
            return
        for directory in chain:
            try:
                resolved = directory.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            directories.append(directory)

    try:
        add_chain(Path.cwd())
    except OSError:
        pass
    add_chain(Path(__file__).resolve().parent)
    return directories


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise ZCodeHeaderContractError(
            "ZCode header contract could not be read"
        ) from None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        raise ZCodeHeaderContractError(
            "ZCode header contract is not valid JSON"
        ) from None
    if not isinstance(payload, dict):
        raise ZCodeHeaderContractError("ZCode header contract must be a JSON object")
    return payload


def _parse_contract(path: Path, payload: Mapping[str, Any]) -> ZCodeHeaderContract:
    schema_version = _require_schema_version(payload)
    _require_client(payload)
    _require_endpoint(payload)
    runtime_header_version, appimage_sha256 = _parse_source_identity(payload)
    openai_token, provider_token = _parse_sdk_tokens(payload)
    return ZCodeHeaderContract(
        schema_version=schema_version,
        runtime_header_version=runtime_header_version,
        appimage_sha256=appimage_sha256,
        profiles=_parse_profiles(payload),
        sdk_openai_compatible_token=openai_token,
        sdk_provider_utils_token=provider_token,
        forbidden_inbound_headers=_parse_forbidden_headers(payload),
        signing=_parse_signing(payload, runtime_header_version),
        source_path=str(path),
    )


def _detach_contract(contract: ZCodeHeaderContract) -> ZCodeHeaderContract:
    return ZCodeHeaderContract(
        schema_version=contract.schema_version,
        runtime_header_version=contract.runtime_header_version,
        appimage_sha256=contract.appimage_sha256,
        profiles={
            name: _copy_string_mapping(headers)
            for name, headers in contract.profiles.items()
        },
        sdk_openai_compatible_token=contract.sdk_openai_compatible_token,
        sdk_provider_utils_token=contract.sdk_provider_utils_token,
        forbidden_inbound_headers=contract.forbidden_inbound_headers,
        signing=contract.signing,
        source_path=contract.source_path,
    )


def _require_schema_version(payload: Mapping[str, Any]) -> int:
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _EXPECTED_SCHEMA_VERSION
    ):
        raise ZCodeHeaderContractError(
            "ZCode header contract field schema_version must be 2"
        )
    return schema_version


def _require_client(payload: Mapping[str, Any]) -> None:
    if payload.get("client") != _EXPECTED_CLIENT:
        raise ZCodeHeaderContractError(
            "ZCode header contract field client must be zcode"
        )


def _require_endpoint(payload: Mapping[str, Any]) -> None:
    endpoint = _require_dict(payload.get("endpoint"), "endpoint")
    if endpoint.get("base_url") != _EXPECTED_ENDPOINT_BASE_URL:
        raise ZCodeHeaderContractError(
            "ZCode header contract field endpoint.base_url is not the coding plan endpoint"
        )


def _parse_source_identity(payload: Mapping[str, Any]) -> tuple[str, str]:
    source = _require_dict(payload.get("source"), "source")
    runtime_header_version = _require_nonempty_str(
        source.get("runtime_header_version"),
        "source.runtime_header_version",
    )
    appimage_sha256 = _require_nonempty_str(
        source.get("appimage_sha256"), "source.appimage_sha256"
    )
    if _SHA256_RE.fullmatch(appimage_sha256) is None:
        raise ZCodeHeaderContractError(
            "ZCode header contract field source.appimage_sha256 must be 64 lowercase hex characters"
        )
    return runtime_header_version, appimage_sha256


def _parse_profiles(payload: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    profiles = _require_dict(payload.get("profiles"), "profiles")
    return {
        "app_server": _parse_profile_headers(profiles, "app_server"),
        "cli": _parse_profile_headers(profiles, "cli"),
    }


def _parse_profile_headers(
    profiles: Mapping[str, Any], profile_name: str
) -> dict[str, str]:
    profile = _require_dict(profiles.get(profile_name), f"profiles.{profile_name}")
    headers = _require_dict(profile.get("headers"), f"profiles.{profile_name}.headers")
    parsed: dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ZCodeHeaderContractError(
                f"ZCode header contract field profiles.{profile_name}.headers must contain string keys and values"
            )
        parsed[key] = value
    return parsed


def _parse_sdk_tokens(payload: Mapping[str, Any]) -> tuple[str, str]:
    assembly = _require_dict(payload.get("model_wire_assembly"), "model_wire_assembly")
    defaults = _require_dict(
        assembly.get("sdk_defaults"), "model_wire_assembly.sdk_defaults"
    )
    user_agent = _require_nonempty_str(
        defaults.get("User-Agent"),
        "model_wire_assembly.sdk_defaults.User-Agent",
    )
    openai_match = _SDK_OPENAI_COMPATIBLE_TOKEN_RE.search(user_agent)
    provider_match = _SDK_PROVIDER_UTILS_TOKEN_RE.search(user_agent)
    if openai_match is None or provider_match is None:
        raise ZCodeHeaderContractError(
            "ZCode header contract is missing AI SDK User-Agent tokens"
        )
    return openai_match.group(0), provider_match.group(0)


def _parse_signing(
    payload: Mapping[str, Any], runtime_header_version: str
) -> ZCodeSigningMaterial:
    signing = _require_dict(payload.get("signing"), "signing")
    feature_gate = _require_dict(signing.get("feature_gate"), "signing.feature_gate")
    handshake = _require_dict(signing.get("handshake"), "signing.handshake")
    return ZCodeSigningMaterial(
        runtime_header_version=runtime_header_version,
        feature_gate_url=_require_nonempty_str(
            feature_gate.get("url"), "signing.feature_gate.url"
        ),
        feature_gate_timeout_seconds=_require_number(
            feature_gate.get("timeout_seconds"),
            "signing.feature_gate.timeout_seconds",
        ),
        feature_gate_cache_ttl_seconds=_require_number(
            feature_gate.get("cache_ttl_seconds"),
            "signing.feature_gate.cache_ttl_seconds",
        ),
        handshake_url=_require_nonempty_str(
            handshake.get("coding_plan_url"),
            "signing.handshake.coding_plan_url",
        ),
        handshake_timeout_seconds=_HANDSHAKE_TIMEOUT_SECONDS,
    )


def _parse_forbidden_headers(payload: Mapping[str, Any]) -> frozenset[str]:
    raw_headers = payload.get("forbidden_inbound_headers")
    if not isinstance(raw_headers, list) or any(
        not isinstance(item, str) for item in raw_headers
    ):
        raise ZCodeHeaderContractError(
            "ZCode header contract field forbidden_inbound_headers must be a list of strings"
        )
    return frozenset(item.lower() for item in raw_headers)


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ZCodeHeaderContractError(
            f"ZCode header contract field {field_name} must be an object"
        )
    return value


def _require_nonempty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ZCodeHeaderContractError(
            f"ZCode header contract field {field_name} must be a non-empty string"
        )
    return value


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ZCodeHeaderContractError(
            f"ZCode header contract field {field_name} must be a number"
        )
    return float(value)


def _session_sources(
    litellm_params: Optional[Mapping[str, Any]],
    optional_params: Optional[Mapping[str, Any]] = None,
) -> tuple[Mapping[str, Any], ...]:
    sources = list(_metadata_dicts(litellm_params))
    if isinstance(optional_params, Mapping):
        sources.append(optional_params)
    if isinstance(litellm_params, Mapping):
        sources.append(litellm_params)
    return tuple(sources)


def _metadata_dicts(
    litellm_params: Optional[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(litellm_params, Mapping):
        return ()
    sources: list[Mapping[str, Any]] = []
    for key in ("metadata", "litellm_metadata"):
        value = litellm_params.get(key)
        if isinstance(value, dict):
            sources.append(value)
    return tuple(sources)


def _first_nonempty_string(
    sources: tuple[Mapping[str, Any], ...], fields: tuple[str, ...]
) -> Optional[str]:
    for field in fields:
        for source in sources:
            value = source.get(field)
            if isinstance(value, str) and value != "":
                return value
    return None


def _select_profile_name(
    profile_name: Optional[str], sources: tuple[Mapping[str, Any], ...]
) -> str:
    if profile_name in _PROFILE_NAMES:
        return str(profile_name)
    metadata_profile = _first_nonempty_string(sources, ("zcode_profile",))
    if metadata_profile in _PROFILE_NAMES:
        return metadata_profile
    if _first_nonempty_string(sources, ("zcode_source_title",)) == "electron":
        return "app_server"
    return "cli"


def _copy_string_mapping(headers: Mapping[str, str]) -> dict[str, str]:
    copied: dict[str, str] = {}
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ZCodeHeaderContractError(
                "selected ZCode profile headers must be strings"
            )
        copied[name] = value
    return copied


def _assemble_user_agent(
    profile_user_agent: str,
    contract: ZCodeHeaderContract,
    runtime_globals: Optional[Mapping[str, Any]],
) -> str:
    return (
        f"{profile_user_agent} {contract.sdk_openai_compatible_token} "
        f"{contract.sdk_provider_utils_token} {_runtime_token(runtime_globals)}"
    )


def _runtime_token(runtime_globals: Optional[Mapping[str, Any]]) -> str:
    globals_map: Mapping[str, Any] = (
        runtime_globals if isinstance(runtime_globals, Mapping) else {}
    )
    if globals_map.get("window"):
        return "runtime/browser"
    user_agent = _mapping_or_attr(globals_map.get("navigator"), "userAgent")
    if isinstance(user_agent, str) and user_agent != "":
        return "runtime/" + user_agent.lower()
    process = globals_map.get("process")
    node = _mapping_or_attr(_mapping_or_attr(process, "versions"), "node")
    version = _mapping_or_attr(process, "version")
    if node and isinstance(version, str):
        return "runtime/node.js/" + version
    if globals_map.get("EdgeRuntime"):
        return "runtime/vercel-edge"
    return "runtime/unknown"


def _mapping_or_attr(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    if value is None:
        return None
    return getattr(value, name, None)


def _attribution_headers(
    sources: tuple[Mapping[str, Any], ...],
    request_id_factory: Optional[Callable[[], str]],
) -> dict[str, str]:
    headers = {
        "x-request-id": _request_id(sources, request_id_factory),
        "x-zcode-trace-id": _trace_id(sources),
        "x-zcode-session-type": _session_type(sources),
    }
    session_id = _optional_prefixed_value(
        sources, _SESSION_ID_FIELDS, _SESSION_PREFIXES
    )
    if session_id is not None:
        headers["x-session-id"] = session_id
    query_id = _optional_prefixed_value(sources, ("query_id",), (_QUERY_PREFIX,))
    if query_id is not None:
        headers["x-query-id"] = query_id
    return headers


def _request_id(
    sources: tuple[Mapping[str, Any], ...],
    request_id_factory: Optional[Callable[[], str]],
) -> str:
    existing = _first_nonempty_string(
        sources, ("request_id", "x-request-id", "litellm_call_id")
    )
    if existing is not None:
        return existing
    if request_id_factory is None:
        return str(uuid.uuid4())
    generated = request_id_factory()
    if isinstance(generated, str) and generated != "":
        return generated
    raise ZCodeHeaderContractError("request_id_factory must return a non-empty string")


def _trace_id(sources: tuple[Mapping[str, Any], ...]) -> str:
    existing = _first_nonempty_string(sources, ("trace_id", "x-zcode-trace-id"))
    if existing is not None:
        return existing
    return str(uuid.uuid4())


def _session_type(sources: tuple[Mapping[str, Any], ...]) -> str:
    existing = _first_nonempty_string(
        sources, ("zcode_session_type", "model_request_session_type")
    )
    if isinstance(existing, str) and existing in _ALLOWED_SESSION_TYPES:
        return existing
    return "other"


def _optional_prefixed_value(
    sources: tuple[Mapping[str, Any], ...],
    fields: tuple[str, ...],
    prefixes: tuple[str, ...],
) -> Optional[str]:
    existing = _first_nonempty_string(sources, fields)
    if existing is None:
        return None
    return _strip_one_prefix(existing, prefixes)


def _strip_one_prefix(value: str, prefixes: tuple[str, ...]) -> str:
    for prefix in prefixes:
        if value.startswith(prefix) and len(value) > len(prefix):
            return value[len(prefix) :]
    return value


def _reject_forbidden_headers(
    headers: Mapping[str, str], forbidden: frozenset[str]
) -> None:
    for name in headers:
        lowered = name.lower()
        if lowered == "authorization":
            continue
        if (
            lowered in forbidden
            or lowered in _EXACT_BLOCKED_HEADER_NAMES
            or lowered.startswith(_BLOCKED_HEADER_PREFIXES)
        ):
            raise ZCodeHeaderContractError(
                "assembled ZCode model headers include a forbidden header name"
            )


__all__ = [
    "ZCODE_DESCRIPTOR_RELATIVE_PATH",
    "ZCODE_HEADER_CONTRACT_PATH_ENV",
    "ZCodeHeaderContract",
    "ZCodeHeaderContractError",
    "ZCodeSigningMaterial",
    "build_zcode_model_headers",
    "load_zcode_header_contract",
    "reset_zcode_header_contract_cache",
    "zcode_profile_headers",
]
