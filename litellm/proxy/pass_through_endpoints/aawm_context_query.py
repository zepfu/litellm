"""Provider-neutral AAWM context cache, query, and asyncpg pool ownership."""

from __future__ import annotations

import asyncio
import importlib
import re
from dataclasses import dataclass
from time import monotonic
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from litellm.secret_managers.main import get_secret_str

SecretGetter = Callable[[str], Optional[str]]
ImportModule = Callable[[str], Any]
MonotonicClock = Callable[[], float]

DynamicCacheKey = tuple[str, str, str, str]
ContextCacheKey = tuple[str, str, str, str, str]
DynamicCacheResult = tuple[bool, Optional[str]]
ContextCacheResult = tuple[bool, Optional[dict[str, str]]]

_AAWM_REFERENCE_IDENTIFIER_LIST_QUERY = """
SELECT DISTINCT rc.name
FROM ag_catalog.raw_content rc
WHERE rc.role = 'reference'
  AND rc.valid_to IS NULL
  AND ($1::text IS NULL OR rc.tenant_id IS NULL OR rc.tenant_id = $1::text)
  AND ($2::text IS NULL OR rc.agent_id IS NULL OR rc.agent_id = $2::text)
  AND rc.name NOT IN (SELECT name FROM public.agents)
  AND rc.name NOT IN (SELECT name || '-instructions' FROM public.agents)
ORDER BY rc.name
"""
_AAWM_SQL_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_AAWM_CONTEXT_GRAB_PROC_NAME_ENV_VARS = (
    "AAWM_CONTEXT_GRAB_PROC_NAME",
    "AAWM_DYNAMIC_CONTEXT_GRAB_PROC_NAME",
)
_AAWM_CONTEXT_GRAB_DEFAULT_PROC_NAME = "tristore_search_exact"
_AAWM_DB_HOST_ENV_VARS = (
    "AAWM_DB_HOST",
    "AAWM_POSTGRES_SERVER",
    "POSTGRES_SERVER",
    "PGHOST",
)
_AAWM_DB_PORT_ENV_VARS = (
    "AAWM_DB_PORT",
    "AAWM_POSTGRES_PORT",
    "POSTGRES_PORT",
    "PGPORT",
)
_AAWM_DB_USER_ENV_VARS = (
    "AAWM_DB_USER",
    "AAWM_POSTGRES_USER",
    "POSTGRES_USER",
    "PGUSER",
)
_AAWM_DB_PASSWORD_ENV_VARS = (
    "AAWM_DB_PASSWORD",
    "AAWM_DB_PWD",
    "AAWM_POSTGRES_PASSWORD",
    "AAWM_POSTGRES_PWD",
    "POSTGRES_PASSWORD",
    "POSTGRES_PWD",
    "PGPASSWORD",
)
_AAWM_DB_NAME_ENV_VARS = (
    "AAWM_DB_NAME",
    "AAWM_POSTGRES_DATABASE",
    "POSTGRES_DATABASE",
    "PGDATABASE",
)
_AAWM_DB_SSLMODE_ENV_VARS = (
    "AAWM_DB_SSLMODE",
    "AAWM_POSTGRES_SSLMODE",
    "POSTGRES_SSLMODE",
    "PGSSLMODE",
)
_AAWM_DB_SSL_BOOL_ENV_VARS = (
    "AAWM_DB_SSL",
    "AAWM_POSTGRES_SSL",
    "POSTGRES_SSL",
)
_AAWM_DB_URL_ENV_VARS = (
    "AAWM_DB_URL",
    "AAWM_DATABASE_URL",
    "AAWM_POSTGRES_URL",
)
_AAWM_DB_APPLICATION_NAME_ENV_VARS = (
    "AAWM_DYNAMIC_INJECTION_DB_APPLICATION_NAME",
    "AAWM_DB_APPLICATION_NAME",
    "AAWM_POSTGRES_APPLICATION_NAME",
    "PGAPPNAME",
)
_AAWM_DYNAMIC_INJECTION_APPLICATION_NAME = "aawm-litellm-dynamic-injection"
_AAWM_DYNAMIC_INJECTION_CACHE_TTL_SECONDS = 15.0
_AAWM_DYNAMIC_INJECTION_POOL_MIN_SIZE = 1
_AAWM_DYNAMIC_INJECTION_POOL_MAX_SIZE = 4
_AAWM_DYNAMIC_INJECTION_COMMAND_TIMEOUT_SECONDS = 10
_AAWM_DYNAMIC_INJECTION_ACQUIRE_TIMEOUT_SECONDS = 10.0
_AAWM_DYNAMIC_INJECTION_STATEMENT_CACHE_SIZE = 0


@dataclass(frozen=True, slots=True)
class ContextQueryRuntime:
    get_secret_str: SecretGetter
    import_module: ImportModule = importlib.import_module
    monotonic: MonotonicClock = monotonic


@dataclass(frozen=True, slots=True)
class ContextQueryServices:
    get_cached_dynamic_result: Callable[[DynamicCacheKey], Awaitable[DynamicCacheResult]]
    set_cached_dynamic_result: Callable[[DynamicCacheKey, Optional[str]], Awaitable[None]]
    get_cached_context_result: Callable[[ContextCacheKey], Awaitable[ContextCacheResult]]
    set_cached_context_result: Callable[[ContextCacheKey, dict[str, str]], Awaitable[None]]
    get_agent_memories: Callable[..., Awaitable[Optional[str]]]
    get_context: Callable[..., Awaitable[Optional[str]]]
    get_reference_identifiers: Callable[..., Awaitable[Optional[str]]]
    get_context_proc_name: Callable[[], str]
    get_context_proc_name_for_logging: Callable[[], str]
    max_parallel_queries: int


_runtime = ContextQueryRuntime(get_secret_str=get_secret_str)
_aawm_dynamic_injection_pool: Optional[Any] = None
_aawm_dynamic_injection_pool_lock = asyncio.Lock()
_aawm_dynamic_injection_cache: dict[DynamicCacheKey, tuple[float, Optional[str]]] = {}
_aawm_dynamic_injection_cache_lock = asyncio.Lock()
_aawm_context_grab_cache: dict[ContextCacheKey, tuple[float, dict[str, str]]] = {}
_aawm_context_grab_cache_lock = asyncio.Lock()


def configure_context_query_runtime(runtime: ContextQueryRuntime) -> None:
    global _runtime
    _runtime = runtime


def build_context_query_services(
    *,
    get_agent_memories: Optional[Callable[..., Awaitable[Optional[str]]]] = None,
    get_context: Optional[Callable[..., Awaitable[Optional[str]]]] = None,
    get_reference_identifiers: Optional[Callable[..., Awaitable[Optional[str]]]] = None,
) -> ContextQueryServices:
    return ContextQueryServices(
        get_cached_dynamic_result=_get_cached_aawm_dynamic_injection_result,
        set_cached_dynamic_result=_set_cached_aawm_dynamic_injection_result,
        get_cached_context_result=_get_cached_aawm_context_grab_result,
        set_cached_context_result=_set_cached_aawm_context_grab_result,
        get_agent_memories=get_agent_memories or _call_aawm_get_agent_memories,
        get_context=get_context or _call_aawm_context_grab,
        get_reference_identifiers=(get_reference_identifiers or _call_aawm_reference_identifier_list),
        get_context_proc_name=_get_aawm_context_grab_proc_name,
        get_context_proc_name_for_logging=(_get_aawm_context_grab_proc_name_for_logging),
        max_parallel_queries=_AAWM_DYNAMIC_INJECTION_POOL_MAX_SIZE,
    )


def _clean_secret_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _get_first_secret_value(secret_names: tuple[str, ...]) -> Optional[str]:
    for secret_name in secret_names:
        value = _clean_secret_string(_runtime.get_secret_str(secret_name))
        if value:
            return value
    return None


def _normalize_aawm_sslmode(value: Optional[str]) -> Optional[str]:
    cleaned = _clean_secret_string(value)
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return "require"
    if lowered in {"0", "false", "no", "off"}:
        return "disable"
    return cleaned


def _get_aawm_dynamic_injection_cache_ttl_seconds() -> float:
    raw_value = _clean_secret_string(_runtime.get_secret_str("AAWM_DYNAMIC_INJECTION_CACHE_TTL_SECONDS"))
    if not raw_value:
        return _AAWM_DYNAMIC_INJECTION_CACHE_TTL_SECONDS
    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        return _AAWM_DYNAMIC_INJECTION_CACHE_TTL_SECONDS


async def _get_cached_aawm_dynamic_injection_result(
    cache_key: DynamicCacheKey,
) -> DynamicCacheResult:
    async with _aawm_dynamic_injection_cache_lock:
        cached_entry = _aawm_dynamic_injection_cache.get(cache_key)
        if cached_entry is None:
            return False, None
        expires_at, cached_value = cached_entry
        if expires_at < _runtime.monotonic():
            _aawm_dynamic_injection_cache.pop(cache_key, None)
            return False, None
        return True, cached_value


async def _set_cached_aawm_dynamic_injection_result(
    cache_key: DynamicCacheKey,
    injected_text: Optional[str],
) -> None:
    ttl_seconds = _get_aawm_dynamic_injection_cache_ttl_seconds()
    if ttl_seconds <= 0:
        return
    async with _aawm_dynamic_injection_cache_lock:
        _aawm_dynamic_injection_cache[cache_key] = (
            _runtime.monotonic() + ttl_seconds,
            injected_text,
        )


async def _get_cached_aawm_context_grab_result(
    cache_key: ContextCacheKey,
) -> ContextCacheResult:
    async with _aawm_context_grab_cache_lock:
        cached_entry = _aawm_context_grab_cache.get(cache_key)
        if cached_entry is None:
            return False, None
        expires_at, cached_value = cached_entry
        if expires_at < _runtime.monotonic():
            _aawm_context_grab_cache.pop(cache_key, None)
            return False, None
        return True, dict(cached_value)


async def _set_cached_aawm_context_grab_result(
    cache_key: ContextCacheKey,
    cached_payload: dict[str, str],
) -> None:
    ttl_seconds = _get_aawm_dynamic_injection_cache_ttl_seconds()
    if ttl_seconds <= 0:
        return
    async with _aawm_context_grab_cache_lock:
        _aawm_context_grab_cache[cache_key] = (
            _runtime.monotonic() + ttl_seconds,
            dict(cached_payload),
        )


def _append_aawm_dynamic_injection_dsn_query_params(
    dsn: str,
    params: dict[str, Optional[str]],
) -> str:
    parsed = urlsplit(dsn)
    if not parsed.scheme:
        return dsn
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    existing_keys = {key for key, _value in query_items}
    for key, value in params.items():
        cleaned_value = _clean_secret_string(value)
        if cleaned_value and key not in existing_keys:
            query_items.append((key, cleaned_value))
            existing_keys.add(key)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query_items),
            parsed.fragment,
        )
    )


def _get_aawm_dynamic_injection_application_name(
    *,
    get_first_secret_value: Optional[Callable[[tuple[str, ...]], Optional[str]]] = None,
) -> str:
    secret_value_getter = get_first_secret_value or _get_first_secret_value
    return secret_value_getter(_AAWM_DB_APPLICATION_NAME_ENV_VARS) or _AAWM_DYNAMIC_INJECTION_APPLICATION_NAME


def _get_aawm_dynamic_injection_server_settings(
    *,
    get_application_name: Optional[Callable[[], str]] = None,
) -> dict[str, str]:
    return {"application_name": (get_application_name or _get_aawm_dynamic_injection_application_name)()}


async def _initialize_aawm_dynamic_injection_connection(
    conn: Any,
    *,
    get_application_name: Optional[Callable[[], str]] = None,
) -> None:
    await conn.execute(
        "select set_config($1, $2, false)",
        "application_name",
        (get_application_name or _get_aawm_dynamic_injection_application_name)(),
    )


def _build_aawm_dynamic_injection_dsn(
    *,
    get_first_secret_value: Optional[Callable[[tuple[str, ...]], Optional[str]]] = None,
    normalize_sslmode: Optional[Callable[[Optional[str]], Optional[str]]] = None,
    get_application_name: Optional[Callable[[], str]] = None,
) -> Optional[str]:
    secret_value_getter = get_first_secret_value or _get_first_secret_value
    sslmode_normalizer = normalize_sslmode or _normalize_aawm_sslmode
    application_name_getter = get_application_name or _get_aawm_dynamic_injection_application_name
    host = secret_value_getter(_AAWM_DB_HOST_ENV_VARS)
    port = secret_value_getter(_AAWM_DB_PORT_ENV_VARS)
    user = secret_value_getter(_AAWM_DB_USER_ENV_VARS)
    password = secret_value_getter(_AAWM_DB_PASSWORD_ENV_VARS)
    database = secret_value_getter(_AAWM_DB_NAME_ENV_VARS)
    sslmode = sslmode_normalizer(
        secret_value_getter(_AAWM_DB_SSLMODE_ENV_VARS) or secret_value_getter(_AAWM_DB_SSL_BOOL_ENV_VARS)
    )
    has_component_config = any((host, port, user, password, database, sslmode))
    if has_component_config:
        if not host or not user or not database:
            return None
        credentials = quote(user, safe="")
        if password:
            credentials += f":{quote(password, safe='')}"
        dsn = f"postgresql://{credentials}@{host}:{port or '5432'}/" f"{quote(database, safe='')}"
        if sslmode:
            dsn += f"?{urlencode({'sslmode': sslmode})}"
        return _append_aawm_dynamic_injection_dsn_query_params(
            dsn,
            {"application_name": application_name_getter()},
        )
    url_dsn = secret_value_getter(_AAWM_DB_URL_ENV_VARS)
    if not url_dsn:
        return None
    return _append_aawm_dynamic_injection_dsn_query_params(
        url_dsn,
        {"application_name": application_name_getter()},
    )


async def _get_aawm_dynamic_injection_pool(
    *,
    build_dsn: Optional[Callable[[], Optional[str]]] = None,
    get_server_settings: Optional[Callable[[], dict[str, str]]] = None,
    initialize_connection: Optional[Callable[[Any], Awaitable[None]]] = None,
) -> Any:
    global _aawm_dynamic_injection_pool
    if _aawm_dynamic_injection_pool is not None:
        return _aawm_dynamic_injection_pool
    async with _aawm_dynamic_injection_pool_lock:
        if _aawm_dynamic_injection_pool is not None:
            return _aawm_dynamic_injection_pool
        dsn = (build_dsn or _build_aawm_dynamic_injection_dsn)()
        if not dsn:
            raise RuntimeError("AAWM dynamic injection database configuration is missing")
        try:
            asyncpg = _runtime.import_module("asyncpg")
        except ModuleNotFoundError as exc:
            raise RuntimeError("AAWM dynamic injection requires asyncpg to be installed") from exc
        _aawm_dynamic_injection_pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=_AAWM_DYNAMIC_INJECTION_POOL_MIN_SIZE,
            max_size=_AAWM_DYNAMIC_INJECTION_POOL_MAX_SIZE,
            command_timeout=_AAWM_DYNAMIC_INJECTION_COMMAND_TIMEOUT_SECONDS,
            statement_cache_size=_AAWM_DYNAMIC_INJECTION_STATEMENT_CACHE_SIZE,
            server_settings=(get_server_settings or _get_aawm_dynamic_injection_server_settings)(),
            init=initialize_connection or _initialize_aawm_dynamic_injection_connection,
        )
        return _aawm_dynamic_injection_pool


async def close_aawm_dynamic_injection_pool() -> None:
    global _aawm_dynamic_injection_pool
    async with _aawm_dynamic_injection_pool_lock:
        pool = _aawm_dynamic_injection_pool
        _aawm_dynamic_injection_pool = None
    if pool is not None:
        await pool.close()


def _aawm_dynamic_injection_acquire_timeout_seconds() -> float:
    raw_value = _clean_secret_string(_runtime.get_secret_str("AAWM_DYNAMIC_INJECTION_ACQUIRE_TIMEOUT_SECONDS"))
    if not raw_value:
        return _AAWM_DYNAMIC_INJECTION_ACQUIRE_TIMEOUT_SECONDS
    try:
        return max(0.1, float(raw_value))
    except (TypeError, ValueError):
        return _AAWM_DYNAMIC_INJECTION_ACQUIRE_TIMEOUT_SECONDS


async def _aawm_pool_fetch(
    pool: Any,
    query: str,
    *args: Any,
    get_timeout: Optional[Callable[[], float]] = None,
) -> Any:
    timeout = (get_timeout or _aawm_dynamic_injection_acquire_timeout_seconds)()
    async with pool.acquire(timeout=timeout) as connection:
        return await connection.fetch(query, *args)


async def _aawm_pool_fetchval(
    pool: Any,
    query: str,
    *args: Any,
    get_timeout: Optional[Callable[[], float]] = None,
) -> Any:
    timeout = (get_timeout or _aawm_dynamic_injection_acquire_timeout_seconds)()
    async with pool.acquire(timeout=timeout) as connection:
        return await connection.fetchval(query, *args)


async def _call_aawm_get_agent_memories(
    *,
    agent_name: str,
    tenant_id: str,
    get_pool: Optional[Callable[[], Awaitable[Any]]] = None,
    pool_fetchval: Optional[Callable[..., Awaitable[Any]]] = None,
) -> Optional[str]:
    pool = await (get_pool or _get_aawm_dynamic_injection_pool)()
    result = await (pool_fetchval or _aawm_pool_fetchval)(
        pool,
        "SELECT get_agent_memories($1, $2)",
        agent_name,
        tenant_id,
    )
    if isinstance(result, str):
        stripped_result = result.strip()
        if stripped_result:
            return stripped_result
    return None


def _get_aawm_context_grab_proc_name() -> str:
    proc_name = _get_first_secret_value(_AAWM_CONTEXT_GRAB_PROC_NAME_ENV_VARS) or _AAWM_CONTEXT_GRAB_DEFAULT_PROC_NAME
    if _AAWM_SQL_IDENTIFIER_PATTERN.fullmatch(proc_name) is None:
        raise RuntimeError("AAWM context grab proc name is invalid")
    return proc_name


def _get_aawm_context_grab_proc_name_for_logging() -> str:
    try:
        return _get_aawm_context_grab_proc_name()
    except Exception:
        return "unknown"


async def _call_aawm_context_grab(
    *,
    name: str,
    tenant_id: Optional[str],
    agent_id: Optional[str],
    get_pool: Optional[Callable[[], Awaitable[Any]]] = None,
    pool_fetch: Optional[Callable[..., Awaitable[Any]]] = None,
    get_proc_name: Optional[Callable[[], str]] = None,
) -> Optional[str]:
    proc_name = (get_proc_name or _get_aawm_context_grab_proc_name)()
    pool = await (get_pool or _get_aawm_dynamic_injection_pool)()
    rows = await (pool_fetch or _aawm_pool_fetch)(
        pool,
        f"SELECT content FROM {proc_name}($1, $2, $3)",
        name,
        tenant_id,
        agent_id,
    )
    contents: list[str] = []
    for row in rows:
        content: Optional[str] = None
        if isinstance(row, dict):
            content = row.get("content")
        elif hasattr(row, "get"):
            content = row.get("content")
        if isinstance(content, str):
            stripped_content = content.strip()
            if stripped_content:
                contents.append(stripped_content)
    if contents:
        return "\n\n".join(contents)
    return None


async def _call_aawm_reference_identifier_list(
    *,
    tenant_id: Optional[str],
    agent_id: Optional[str],
    get_pool: Optional[Callable[[], Awaitable[Any]]] = None,
    pool_fetch: Optional[Callable[..., Awaitable[Any]]] = None,
) -> Optional[str]:
    pool = await (get_pool or _get_aawm_dynamic_injection_pool)()
    rows = await (pool_fetch or _aawm_pool_fetch)(
        pool,
        _AAWM_REFERENCE_IDENTIFIER_LIST_QUERY,
        tenant_id,
        agent_id,
    )
    identifier_names: list[str] = []
    for row in rows:
        identifier_name: Optional[str] = None
        if isinstance(row, dict):
            identifier_name = row.get("name")
        elif hasattr(row, "get"):
            identifier_name = row.get("name")
        if isinstance(identifier_name, str):
            stripped_identifier_name = identifier_name.strip()
            if stripped_identifier_name:
                identifier_names.append(stripped_identifier_name)
    if identifier_names:
        return ", ".join(identifier_names)
    return None
