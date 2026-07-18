"""Google Code Assist process-cache ownership and orchestration."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Callable, MutableMapping, NoReturn, Optional, Protocol, TypeVar

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload

_GOOGLE_ADAPTER_TOKEN_CACHE_MAX_SIZE = 256

_google_code_assist_project_cache: dict[str, str] = {}
_google_code_assist_project_lock = asyncio.Lock()
_google_code_assist_prime_until_monotonic_by_key: dict[str, float] = {}
_google_code_assist_prime_quota_by_key: dict[str, Payload] = {}
_google_code_assist_prime_lock = asyncio.Lock()
_google_adapter_semaphores: dict[tuple[str, int], asyncio.Semaphore] = {}
_google_adapter_user_prompt_turn_lock = asyncio.Lock()
_google_adapter_user_prompt_turn_counters: dict[str, int] = {}
_codex_google_code_assist_tool_call_name_cache: dict[str, tuple[str, float]] = {}
_codex_google_code_assist_tool_call_arguments_cache: dict[str, tuple[str, float]] = {}

_K = TypeVar("_K")
_V = TypeVar("_V")


class HttpResponse(Protocol):
    """Response surface required by Code Assist cache orchestration."""

    @property
    def status_code(self) -> int:
        ...

    @property
    def text(self) -> str:
        ...

    @property
    def content(self) -> bytes:
        ...

    def json(self) -> object:
        ...


class BuildHeaders(Protocol):
    def __call__(
        self,
        *,
        adapter_provider: str,
        access_token: str,
        model: Optional[str],
        accept: str,
    ) -> dict[str, str]:
        ...


class ValidateEgress(Protocol):
    def __call__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        credential_family: str,
        expected_target_family: str,
    ) -> object:
        ...


class PostJson(Protocol):
    async def __call__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: Payload,
        timeout: float,
    ) -> HttpResponse:
        ...


class CaptureShape(Protocol):
    def __call__(
        self,
        *,
        mode: str,
        provider: str,
        url_route: str,
        request_body: Payload,
        response: HttpResponse,
        response_body: object,
        response_content: bytes,
        extra_metadata: Payload,
    ) -> object:
        ...


class RaiseHttpError(Protocol):
    def __call__(self, *, status_code: int, detail: str) -> NoReturn:
        ...


class GetRateLimitKey(Protocol):
    def __call__(
        self,
        model: Optional[str],
        *,
        access_token: Optional[str],
        companion_project: Optional[str],
    ) -> str:
        ...


@dataclass(frozen=True)
class Runtime:
    """Injected network, policy, and observability services."""

    get_target_base: Callable[[str], str]
    build_headers: BuildHeaders
    validate_egress: ValidateEgress
    post_json: PostJson
    capture_shape: CaptureShape
    clean_value: Callable[[object], Optional[str]]
    raise_http_error: RaiseHttpError
    get_prime_ttl_seconds: Callable[[], float]
    get_prime_cache_key: Callable[[str, str], str]
    sanitize_quota: Callable[[object, str], Optional[Payload]]
    get_max_concurrent: Callable[[], int]
    get_rate_limit_key: GetRateLimitKey
    monotonic: Callable[[], float]
    debug_enabled: Callable[[], bool]
    log_info: Callable[[str, object], object]
    default_provider: str = "gemini"


def _bound_google_adapter_token_cache(
    cache: MutableMapping[_K, _V],
    *,
    max_size: int = _GOOGLE_ADAPTER_TOKEN_CACHE_MAX_SIZE,
) -> None:
    """Bound insertion-ordered token-derived process state."""
    while len(cache) > max_size:
        try:
            oldest = next(iter(cache))
        except StopIteration:
            break
        cache.pop(oldest, None)


async def _get_or_load_google_code_assist_project(
    access_token: str,
    *,
    runtime: Runtime,
    adapter_provider: Optional[str] = None,
) -> str:
    """Return a cached companion project or load it once per credential."""
    provider = adapter_provider or runtime.default_provider
    target_base = runtime.get_target_base(provider)
    cache_key = hashlib.sha256(f"{provider}:{target_base}:{access_token}".encode("utf-8")).hexdigest()
    cached_project = _google_code_assist_project_cache.get(cache_key)
    if cached_project:
        return cached_project

    async with _google_code_assist_project_lock:
        cached_project = _google_code_assist_project_cache.get(cache_key)
        if cached_project:
            return cached_project

        load_url = f"{target_base.rstrip('/')}/v1internal:loadCodeAssist"
        request_body: Payload = {
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }
        }
        headers = runtime.build_headers(
            adapter_provider=provider,
            access_token=access_token,
            model=None,
            accept="application/json",
        )
        runtime.validate_egress(
            url=load_url,
            headers=headers,
            credential_family="google",
            expected_target_family="google",
        )
        response = await runtime.post_json(
            url=load_url,
            headers=headers,
            body=request_body,
            timeout=30.0,
        )
        try:
            response_body = response.json()
        except Exception:
            response_body = None
        runtime.capture_shape(
            mode="google_code_assist_loadCodeAssist",
            provider=provider,
            url_route=load_url,
            request_body=request_body,
            response=response,
            response_body=response_body,
            response_content=response.content,
            extra_metadata={
                "direct_google_code_assist_preflight": True,
                "code_assist_adapter_provider": provider,
            },
        )
        if response.status_code != 200:
            runtime.raise_http_error(
                status_code=500,
                detail=("Failed to load Google Code Assist project for Anthropic " f"adapter models: {response.text}"),
            )

        if not isinstance(response_body, dict):
            response_body = response.json()
        if not isinstance(response_body, dict):
            runtime.raise_http_error(
                status_code=500,
                detail="Google Code Assist bootstrap returned a non-object response.",
            )
        project = runtime.clean_value(response_body.get("cloudaicompanionProject"))
        if project is None:
            runtime.raise_http_error(
                status_code=500,
                detail=("Google Code Assist bootstrap did not return a " "cloudaicompanionProject."),
            )
            raise RuntimeError("raise_http_error returned unexpectedly")

        _google_code_assist_project_cache[cache_key] = project
        _bound_google_adapter_token_cache(_google_code_assist_project_cache)
        return project


def _get_google_adapter_semaphore(
    model: Optional[str] = None,
    *,
    runtime: Runtime,
    access_token: Optional[str] = None,
    companion_project: Optional[str] = None,
    rate_limit_key: Optional[str] = None,
) -> asyncio.Semaphore:
    """Return the bounded semaphore for a credential/model lane."""
    max_concurrent = runtime.get_max_concurrent()
    resolved_rate_limit_key = runtime.clean_value(rate_limit_key) or runtime.get_rate_limit_key(
        model,
        access_token=access_token,
        companion_project=companion_project,
    )
    semaphore_key = (resolved_rate_limit_key, max_concurrent)
    semaphore = _google_adapter_semaphores.get(semaphore_key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(max_concurrent)
        _google_adapter_semaphores[semaphore_key] = semaphore
        _bound_google_adapter_token_cache(_google_adapter_semaphores)
    return semaphore


async def _prime_google_code_assist_session(
    access_token: str,
    companion_project: str,
    *,
    runtime: Runtime,
    adapter_provider: Optional[str] = None,
) -> Optional[Payload]:
    """Run bounded Code Assist preflights and cache sanitized quota metadata."""
    provider = adapter_provider or runtime.default_provider
    ttl_seconds = runtime.get_prime_ttl_seconds()
    token_key = f"{provider}:{access_token}" if provider != runtime.default_provider else access_token
    cache_key = runtime.get_prime_cache_key(token_key, companion_project)
    async with _google_code_assist_prime_lock:
        if ttl_seconds > 0:
            cached_until = _google_code_assist_prime_until_monotonic_by_key.get(cache_key, 0.0)
            if cached_until > runtime.monotonic():
                if runtime.debug_enabled():
                    runtime.log_info(
                        "Google adapter prime cache hit for project=%s",
                        companion_project,
                    )
                return _google_code_assist_prime_quota_by_key.get(cache_key)

        target_base = runtime.get_target_base(provider).rstrip("/")
        headers = runtime.build_headers(
            adapter_provider=provider,
            access_token=access_token,
            model=None,
            accept="application/json",
        )
        metadata: Payload = {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
            "duetProject": companion_project,
        }
        preflight_requests: tuple[tuple[str, Payload], ...] = (
            (
                f"{target_base}/v1internal:retrieveUserQuota",
                {"project": companion_project},
            ),
            (
                f"{target_base}/v1internal:fetchAdminControls",
                {"project": companion_project},
            ),
            (
                f"{target_base}/v1internal:listExperiments",
                {"project": companion_project, "metadata": metadata},
            ),
        )
        sanitized_quota_response: Optional[Payload] = None
        for url, body in preflight_requests:
            runtime.validate_egress(
                url=url,
                headers=headers,
                credential_family="google",
                expected_target_family="google",
            )
            try:
                response = await runtime.post_json(
                    url=url,
                    headers=headers,
                    body=body,
                    timeout=20.0,
                )
            except Exception:
                continue
            try:
                response_body = response.json()
            except Exception:
                response_body = None
            runtime.capture_shape(
                mode="google_code_assist_preflight",
                provider=provider,
                url_route=url,
                request_body=body,
                response=response,
                response_body=response_body,
                response_content=response.content,
                extra_metadata={
                    "direct_google_code_assist_preflight": True,
                    "code_assist_adapter_provider": provider,
                    "preflight_endpoint": url.rsplit(":", 1)[-1],
                },
            )
            if "retrieveUserQuota" in url:
                source = (
                    "antigravity_retrieve_user_quota" if provider == "antigravity" else "google_retrieve_user_quota"
                )
                sanitized_quota_response = runtime.sanitize_quota(response_body, source)

        if ttl_seconds > 0:
            _google_code_assist_prime_until_monotonic_by_key[cache_key] = runtime.monotonic() + ttl_seconds
        if sanitized_quota_response:
            _google_code_assist_prime_quota_by_key[cache_key] = sanitized_quota_response
            _bound_google_adapter_token_cache(_google_code_assist_prime_until_monotonic_by_key)
            _bound_google_adapter_token_cache(_google_code_assist_prime_quota_by_key)
        return sanitized_quota_response


__all__ = [
    "Runtime",
    "_bound_google_adapter_token_cache",
    "_get_google_adapter_semaphore",
    "_get_or_load_google_code_assist_project",
    "_prime_google_code_assist_session",
]
