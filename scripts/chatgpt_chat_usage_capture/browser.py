"""Read-only Playwright transport for a dedicated ChatGPT browser profile."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .adapter import (
    ALLOWED_METHODS,
    ALLOWED_PATH_PREFIXES,
    AdapterError,
    ChatGPTHistoryAdapter,
)
from .config import AccountConfig

LIVE_GATE = (
    "live Playwright collection requires Playwright and an existing dedicated browser "
    "profile. Tokens and cookies remain in that profile and are never copied into SQLite."
)
CHATGPT_ORIGIN = "https://chatgpt.com"


class LiveBrowserUnavailable(AdapterError):
    """Raised when live browser collection cannot start honestly."""


def playwright_available() -> bool:
    try:
        import playwright  # noqa: F401
    except Exception:
        return False
    return True


def dedicated_profile_ready(account: AccountConfig) -> bool:
    path = account.browser.profile_path
    if not path:
        return False
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return False
    if resolved == Path.home().resolve():
        return False
    if resolved.name.lower() in {"browser", "default", "chrome", "chromium"}:
        return False
    return resolved.exists() and resolved.is_dir()


def live_browser_gate(account: AccountConfig) -> str | None:
    if not playwright_available():
        return "playwright_unavailable"
    if not dedicated_profile_ready(account):
        if not account.browser.allow_interactive_login:
            return "interactive_login_disabled"
        return "dedicated_profile_missing"
    return None


def build_playwright_adapter(account: AccountConfig) -> ChatGPTHistoryAdapter:
    reason = live_browser_gate(account)
    if reason is not None:
        raise LiveBrowserUnavailable(f"{LIVE_GATE} ({reason})")
    transport = PlaywrightTransport(account)
    return ChatGPTHistoryAdapter(
        transport,
        expected_identity={
            "provider_user_id": account.expected_provider_user_id,
            "workspace_id": account.expected_workspace_id,
        },
    )


class PlaywrightTransport:
    """GET-only allowlisted transport using a dedicated persistent Playwright context."""

    def __init__(self, account: AccountConfig) -> None:
        self.account = account
        self.requests: list[dict[str, Any]] = []
        self._playwright: Any = None
        self._context: Any = None
        self._request_context: Any = None

    def request(self, method: str, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        method = method.upper()
        if method not in ALLOWED_METHODS:
            raise AdapterError(f"method not allowlisted: {method} {path}")
        if not _is_allowed_path(path):
            raise AdapterError(f"path not allowlisted: {path}")
        query = dict(params or {})
        self.requests.append({"method": method, "path": path, "params": query})
        request_context = self._ensure_request_context()
        try:
            response = request_context.get(
                f"{CHATGPT_ORIGIN}{path}",
                params=query,
                timeout=self.account.collection.request_timeout_seconds * 1000,
            )
        except Exception as exc:
            raise AdapterError(f"browser GET failed for {path}") from exc
        return _adapt_response(response)

    def close(self) -> None:
        context = self._context
        playwright = self._playwright
        self._request_context = None
        self._context = None
        self._playwright = None
        if context is not None:
            context.close()
        if playwright is not None:
            playwright.stop()

    def _ensure_request_context(self) -> Any:
        if self._request_context is not None:
            return self._request_context
        reason = live_browser_gate(self.account)
        if reason is not None:
            raise LiveBrowserUnavailable(f"{LIVE_GATE} ({reason})")
        try:
            from playwright.sync_api import sync_playwright

            self._playwright = sync_playwright().start()
            self._context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.account.browser.profile_path),
                headless=self.account.browser.headless,
                accept_downloads=False,
            )
            self._request_context = self._context.request
        except Exception as exc:
            self.close()
            raise LiveBrowserUnavailable(f"{LIVE_GATE} (launch_failed)") from exc
        return self._request_context


def _is_allowed_path(path: str) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in ALLOWED_PATH_PREFIXES)


def _adapt_response(response: Any) -> dict[str, Any]:
    status = int(response.status)
    headers = response.headers
    content_type = str(headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    retry_after = headers.get("retry-after")
    try:
        payload = response.json()
    except Exception:
        payload = {}
    if not isinstance(payload, Mapping):
        payload = {}
    adapted = dict(payload)
    adapted["http_status"] = status
    if content_type:
        adapted["content_type"] = content_type
    if retry_after is not None:
        adapted["retry_after"] = str(retry_after)
    return adapted
