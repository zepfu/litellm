"""YAML configuration for the standalone ChatGPT Chat usage collector."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Optional

from .privacy import SURFACE_CHAT, classify_surface
from .timeutil import DurationError, parse_iso_duration

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - pyyaml is pinned in this repo
    raise RuntimeError("pyyaml is required to load chatgpt-chat-usage-capture config") from exc


DEFAULT_REFRESH_INTERVAL = "PT1H"
DEFAULT_OVERLAP = "PT48H"
DEFAULT_BACKFILL = "P14D"
DEFAULT_LOOKBACK = "P7D"
DEFAULT_TIMEZONE = "America/New_York"
MINIMUM_REFRESH = timedelta(minutes=5)
CANONICAL_FAMILIES = ("astra_pro", "sol_pro", "other_chat", "unknown")
WINDOW_TYPES = (
    "provider_explicit",
    "operator_explicit",
    "anchored_elapsed",
    "calendar",
    "rolling_elapsed",
    "unknown",
)


class ConfigError(ValueError):
    """Invalid collector configuration."""


@dataclass(frozen=True)
class BrowserConfig:
    adapter: str
    profile_path: Path
    headless: bool
    allow_interactive_login: bool


@dataclass(frozen=True)
class SchedulerConfig:
    refresh_interval: timedelta
    reconciliation_interval: timedelta
    jitter_seconds: int
    catch_up: str
    refresh_on_startup: str
    overlap_duration: timedelta
    initial_backfill_duration: timedelta
    reconciliation_lookback: timedelta


@dataclass(frozen=True)
class CollectionConfig:
    include_archived: bool
    include_projects: bool
    require_surface_evidence: bool
    request_gap_ms: int
    request_timeout_seconds: int
    max_http_attempts_per_run: int
    max_run_seconds: int
    max_pages_per_conversation_per_run: int
    observe_quota_metadata: bool


@dataclass(frozen=True)
class AccountingConfig:
    working_estimator: str
    count_only_generation_started_or_completed: bool
    uncertain_attempts_in_working_estimate: bool
    include_unknown_surface_in_chat: bool
    missing_quota_window_behavior: str
    boundary_convention: str


@dataclass(frozen=True)
class QuotaWindow:
    type: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    timezone: Optional[str] = None
    duration: Optional[timedelta] = None
    evidence: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class QuotaBucket:
    id: str
    families: tuple[str, ...]
    capacity: Optional[int]
    unit: str
    documented_period_hint: Optional[str]
    window: QuotaWindow
    membership: str = "union_once_per_attempt"


@dataclass(frozen=True)
class QuotaPolicy:
    id: str
    status: str
    surface: str
    buckets: tuple[QuotaBucket, ...]


@dataclass(frozen=True)
class ModelMapping:
    version: str
    canonical_families: tuple[str, ...]
    exact_rules: tuple[dict[str, Any], ...]
    unknown_behavior: str


@dataclass(frozen=True)
class AccountConfig:
    id: str
    enabled: bool
    provider: str
    expected_provider_user_id: Optional[str]
    expected_workspace_id: Optional[str]
    quota_owner_id: str
    surface: str
    plan_policy_id: str
    browser: BrowserConfig
    scheduler: SchedulerConfig
    collection: CollectionConfig


@dataclass(frozen=True)
class ApplicationConfig:
    name: str
    report_timezone: str
    database_path: Path
    persist_message_content: bool
    bind_host: str
    bind_port: int


@dataclass(frozen=True)
class CollectorConfig:
    schema_version: int
    application: ApplicationConfig
    accounts: tuple[AccountConfig, ...]
    accounting: AccountingConfig
    model_mapping: ModelMapping
    quota_policies: tuple[QuotaPolicy, ...]
    default_lookback: timedelta
    source_path: Path

    def account(self, account_id: Optional[str] = None) -> AccountConfig:
        if account_id is None:
            enabled = [item for item in self.accounts if item.enabled]
            if len(enabled) != 1 and len(self.accounts) != 1:
                raise ConfigError("select --account; multiple accounts are configured")
            return enabled[0] if enabled else self.accounts[0]
        for item in self.accounts:
            if item.id == account_id:
                return item
        raise ConfigError(f"unknown account: {account_id}")

    def policy_for(self, account: AccountConfig) -> QuotaPolicy:
        for policy in self.quota_policies:
            if policy.id == account.plan_policy_id:
                return policy
        raise ConfigError(f"unknown quota policy: {account.plan_policy_id}")


def load_config(path: str | Path) -> CollectorConfig:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ConfigError("config root must be a mapping")
    return parse_config(payload, source_path=config_path)


def parse_config(payload: Mapping[str, Any], *, source_path: Path) -> CollectorConfig:
    schema_version = int(payload.get("schema_version") or 1)
    application = _parse_application(payload.get("application") or {}, source_path)
    accounts = tuple(
        _parse_account(item, source_path) for item in (payload.get("accounts") or [])
    )
    if not accounts:
        raise ConfigError("at least one account is required")
    enabled = [item for item in accounts if item.enabled]
    if len(enabled) > 1:
        raise ConfigError("enable only one account by default")
    accounting = _parse_accounting(payload.get("accounting") or {})
    mapping = _parse_mapping(payload.get("model_mapping") or {})
    policies = tuple(_parse_policy(item) for item in (payload.get("quota_policies") or []))
    reporting = payload.get("reporting") or {}
    lookback = parse_iso_duration(str(reporting.get("default_lookback") or DEFAULT_LOOKBACK))
    _validate_retention(payload.get("retention") or {}, accounts)
    return CollectorConfig(
        schema_version=schema_version,
        application=application,
        accounts=accounts,
        accounting=accounting,
        model_mapping=mapping,
        quota_policies=policies,
        default_lookback=lookback,
        source_path=source_path,
    )


def _parse_application(raw: Mapping[str, Any], source_path: Path) -> ApplicationConfig:
    database = Path(str(raw.get("database_path") or "./state/usage.sqlite"))
    if not database.is_absolute():
        database = (source_path.parent / database).resolve()
    persist = bool(raw.get("persist_message_content") or False)
    if persist:
        raise ConfigError("persist_message_content must remain false")
    return ApplicationConfig(
        name=str(raw.get("name") or "chatgpt-chat-usage-capture"),
        report_timezone=str(raw.get("report_timezone") or DEFAULT_TIMEZONE),
        database_path=database,
        persist_message_content=False,
        bind_host=str(raw.get("bind_host") or "127.0.0.1"),
        bind_port=int(raw.get("bind_port") or 8765),
    )


def _parse_account(raw: Mapping[str, Any], source_path: Path) -> AccountConfig:
    account_id = str(raw.get("id") or "").strip()
    if not account_id:
        raise ConfigError("account.id is required")
    surface = classify_surface(raw.get("surface") or SURFACE_CHAT, default=None)
    if surface != SURFACE_CHAT:
        raise ConfigError(f"account {account_id} must set surface=chat")
    browser_raw = raw.get("browser") or {}
    profile = Path(str(browser_raw.get("profile_path") or f"./state/browser/{account_id}"))
    if not profile.is_absolute():
        profile = (source_path.parent / profile).resolve()
    scheduler_raw = raw.get("scheduler") or {}
    collection_raw = raw.get("collection") or {}
    return AccountConfig(
        id=account_id,
        enabled=bool(raw.get("enabled", True)),
        provider=str(raw.get("provider") or "openai"),
        expected_provider_user_id=_optional_str(raw.get("expected_provider_user_id")),
        expected_workspace_id=_optional_str(raw.get("expected_workspace_id")),
        quota_owner_id=str(raw.get("quota_owner_id") or account_id),
        surface=SURFACE_CHAT,
        plan_policy_id=str(raw.get("plan_policy_id") or "pro200-chat-2026-09-05"),
        browser=BrowserConfig(
            adapter=str(browser_raw.get("adapter") or "fixture_history"),
            profile_path=profile,
            headless=bool(browser_raw.get("headless") or False),
            allow_interactive_login=bool(browser_raw.get("allow_interactive_login", True)),
        ),
        scheduler=_parse_scheduler(scheduler_raw),
        collection=CollectionConfig(
            include_archived=bool(collection_raw.get("include_archived", True)),
            include_projects=bool(collection_raw.get("include_projects", True)),
            require_surface_evidence=bool(collection_raw.get("require_surface_evidence", True)),
            request_gap_ms=int(collection_raw.get("request_gap_ms") or 1000),
            request_timeout_seconds=int(collection_raw.get("request_timeout_seconds") or 30),
            max_http_attempts_per_run=int(collection_raw.get("max_http_attempts_per_run") or 500),
            max_run_seconds=int(collection_raw.get("max_run_seconds") or 1200),
            max_pages_per_conversation_per_run=int(
                collection_raw.get("max_pages_per_conversation_per_run") or 100
            ),
            observe_quota_metadata=bool(collection_raw.get("observe_quota_metadata", True)),
        ),
    )


def _parse_scheduler(raw: Mapping[str, Any]) -> SchedulerConfig:
    refresh = parse_iso_duration(str(raw.get("refresh_interval") or DEFAULT_REFRESH_INTERVAL))
    if refresh < MINIMUM_REFRESH:
        raise ConfigError("refresh_interval must be at least PT5M")
    try:
        overlap = parse_iso_duration(str(raw.get("overlap_duration") or DEFAULT_OVERLAP))
        backfill = parse_iso_duration(str(raw.get("initial_backfill_duration") or DEFAULT_BACKFILL))
        recon_lookback = parse_iso_duration(
            str(raw.get("reconciliation_lookback") or DEFAULT_BACKFILL)
        )
        recon = parse_iso_duration(str(raw.get("reconciliation_interval") or "PT24H"))
    except DurationError as exc:
        raise ConfigError(str(exc)) from exc
    catch_up = str(raw.get("catch_up") or "coalesce")
    if catch_up != "coalesce":
        raise ConfigError("scheduler.catch_up must be coalesce")
    return SchedulerConfig(
        refresh_interval=refresh,
        reconciliation_interval=recon,
        jitter_seconds=int(raw.get("jitter_seconds") or 60),
        catch_up=catch_up,
        refresh_on_startup=str(raw.get("refresh_on_startup") or "if_due"),
        overlap_duration=overlap,
        initial_backfill_duration=backfill,
        reconciliation_lookback=recon_lookback,
    )


def _parse_accounting(raw: Mapping[str, Any]) -> AccountingConfig:
    return AccountingConfig(
        working_estimator=str(
            raw.get("working_estimator") or "requested_if_known_else_recorded_final"
        ),
        count_only_generation_started_or_completed=bool(
            raw.get("count_only_generation_started_or_completed", True)
        ),
        uncertain_attempts_in_working_estimate=bool(
            raw.get("uncertain_attempts_in_working_estimate") or False
        ),
        include_unknown_surface_in_chat=bool(
            raw.get("include_unknown_surface_in_chat") or False
        ),
        missing_quota_window_behavior=str(
            raw.get("missing_quota_window_behavior") or "show_activity_only"
        ),
        boundary_convention=str(
            raw.get("boundary_convention") or "start_inclusive_end_exclusive"
        ),
    )


def _parse_mapping(raw: Mapping[str, Any]) -> ModelMapping:
    families = tuple(raw.get("canonical_families") or CANONICAL_FAMILIES)
    rules = tuple(item for item in (raw.get("exact_rules") or []) if isinstance(item, Mapping))
    return ModelMapping(
        version=str(raw.get("version") or "initial-unmapped"),
        canonical_families=families,
        exact_rules=tuple(dict(item) for item in rules),
        unknown_behavior=str(raw.get("unknown_behavior") or "preserve_and_report"),
    )


def _parse_policy(raw: Mapping[str, Any]) -> QuotaPolicy:
    policy_id = str(raw.get("id") or "").strip()
    if not policy_id:
        raise ConfigError("quota policy id is required")
    surface = classify_surface(raw.get("surface") or SURFACE_CHAT, default=None)
    if surface != SURFACE_CHAT:
        raise ConfigError(f"quota policy {policy_id} must remain surface=chat")
    buckets = tuple(_parse_bucket(item) for item in (raw.get("buckets") or []))
    return QuotaPolicy(
        id=policy_id,
        status=str(raw.get("status") or "documented_seed_requires_account_verification"),
        surface=SURFACE_CHAT,
        buckets=buckets,
    )


def _parse_bucket(raw: Mapping[str, Any]) -> QuotaBucket:
    window_raw = raw.get("window") or {"type": "unknown"}
    window_type = str(window_raw.get("type") or "unknown")
    if window_type not in WINDOW_TYPES:
        raise ConfigError(f"unsupported window type: {window_type}")
    duration = None
    if window_raw.get("duration"):
        duration = parse_iso_duration(str(window_raw["duration"]))
    start = _optional_datetime(window_raw.get("start"))
    end = _optional_datetime(window_raw.get("end"))
    return QuotaBucket(
        id=str(raw.get("id") or "").strip(),
        families=tuple(raw.get("families") or ()),
        capacity=None if raw.get("capacity") is None else int(raw["capacity"]),
        unit=str(raw.get("unit") or "message"),
        documented_period_hint=_optional_str(raw.get("documented_period_hint")),
        membership=str(raw.get("membership") or "union_once_per_attempt"),
        window=QuotaWindow(
            type=window_type,
            start=start,
            end=end,
            timezone=_optional_str(window_raw.get("timezone")),
            duration=duration,
            evidence=_optional_str(window_raw.get("evidence")),
            reason=_optional_str(window_raw.get("reason")),
        ),
    )


def _validate_retention(raw: Mapping[str, Any], accounts: tuple[AccountConfig, ...]) -> None:
    observation_days = int(raw.get("sanitized_observation_days") or 45)
    attempt_days = int(raw.get("normalized_attempt_days") or 180)
    if observation_days < 45 or attempt_days < 180:
        raise ConfigError("retention is shorter than the collector defaults")
    max_backfill = max(item.scheduler.initial_backfill_duration for item in accounts)
    if timedelta(days=observation_days) < max_backfill:
        raise ConfigError("observation retention is shorter than the configured backfill")


def _optional_str(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


def _optional_datetime(value: Any):
    if value is None or value == "":
        return None
    from .timeutil import parse_datetime

    return parse_datetime(value)
