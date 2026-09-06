"""ChatGPT ordinary Chat usage collector CLI (D1-752)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .accounting import record_manual_observation, rebuild_aggregates, set_explicit_window
from .collector import Collector
from .config import CollectorConfig, ConfigError, load_config
from .ledger import Ledger
from .reporting import build_report
from .scheduler import LeaseHeldError, Scheduler
from .timeutil import ensure_utc, isoformat_utc, parse_datetime, parse_iso_duration


def _ensure_ledger_path(config: CollectorConfig) -> Path:
    path = config.application.database_path
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _print(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


def _parse_since(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    raw = value.strip()
    if raw.endswith(("d", "h", "m")) and raw[:-1].isdigit():
        unit = raw[-1]
        amount = int(raw[:-1])
        if unit == "d":
            return ensure_utc(datetime.now(timezone.utc)) - parse_iso_duration(f"P{amount}D")
        if unit == "h":
            return ensure_utc(datetime.now(timezone.utc)) - parse_iso_duration(f"PT{amount}H")
        return ensure_utc(datetime.now(timezone.utc)) - parse_iso_duration(f"PT{amount}M")
    if raw.startswith("P"):
        return ensure_utc(datetime.now(timezone.utc)) - parse_iso_duration(raw)
    parsed = parse_datetime(raw)
    if parsed is None:
        raise ValueError(f"unrecognized since/from value: {value}")
    return parsed


def _lookback(value: Optional[str]):
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("d") and raw[:-1].isdigit():
        return parse_iso_duration(f"P{int(raw[:-1])}D")
    if raw.endswith("h") and raw[:-1].isdigit():
        return parse_iso_duration(f"PT{int(raw[:-1])}H")
    if raw.startswith("P"):
        return parse_iso_duration(raw)
    return None


def cmd_init(config: CollectorConfig, args: argparse.Namespace) -> int:
    path = _ensure_ledger_path(config)
    ledger = Ledger(path)
    try:
        for account in config.accounts:
            ledger.upsert_account(
                {
                    "collector_account_id": account.id,
                    "provider_user_id": account.expected_provider_user_id,
                    "workspace_id": account.expected_workspace_id,
                    "quota_owner_id": account.quota_owner_id,
                    "surface": account.surface,
                    "auth_state": "unconfigured",
                    "plan_policy_id": account.plan_policy_id,
                    "enabled": account.enabled,
                    "profile_path": str(account.browser.profile_path),
                }
            )
        _print(
            {
                "status": "initialized",
                "database_path": str(path),
                "accounts": [account.id for account in config.accounts],
            }
        )
        return 0
    finally:
        ledger.close()


def cmd_browser_login(config: CollectorConfig, args: argparse.Namespace) -> int:
    account = config.account(args.account)
    _print(
        {
            "status": "interactive_required",
            "account_id": account.id,
            "profile_path": str(account.browser.profile_path),
            "note": "Use a user-authorized browser session. This collector never logs in by itself.",
        }
    )
    return 0


def cmd_inspect(config: CollectorConfig, args: argparse.Namespace) -> int:
    ledger = Ledger(_ensure_ledger_path(config))
    try:
        collector = Collector(config, ledger, fixture_root=args.fixture_root)
        result = collector.inspect_capabilities(args.account)
        _print(result)
        return 0
    finally:
        ledger.close()


def _run_collect(config: CollectorConfig, args: argparse.Namespace, mode: str, *, force: bool = False) -> int:
    ledger = Ledger(_ensure_ledger_path(config))
    try:
        collector = Collector(config, ledger, fixture_root=args.fixture_root)
        scheduler = Scheduler(config, ledger, collector)
        since = _parse_since(getattr(args, "since", None))
        try:
            result = scheduler.run_if_due(
                account_id=args.account,
                force=force,
                mode=mode,
                since=since,
            )
        except LeaseHeldError as exc:
            _print({"status": "coalesced", "reason": str(exc)})
            return 0
        if result is None:
            _print({"status": "not_due"})
            return 0
        _print(
            {
                "run_id": result.run_id,
                "mode": result.mode,
                "result": result.result,
                "coverage": result.coverage,
                "new_attempts": result.new_attempts,
                "updated_attempts": result.updated_attempts,
                "deduplicated_attempts": result.deduplicated_attempts,
                "conversations_seen": result.conversations_seen,
                "pages_fetched": result.pages_fetched,
                "missed_intervals": result.missed_intervals,
                "warnings": result.warnings[:20],
                "requests": len(result.requests),
            }
        )
        return 0 if result.result == "complete" else 1
    finally:
        ledger.close()


def cmd_collect(config: CollectorConfig, args: argparse.Namespace) -> int:
    return _run_collect(config, args, args.mode or "refresh", force=True)


def cmd_backfill(config: CollectorConfig, args: argparse.Namespace) -> int:
    return _run_collect(config, args, "backfill", force=True)


def cmd_refresh(config: CollectorConfig, args: argparse.Namespace) -> int:
    return _run_collect(config, args, "refresh", force=True)


def cmd_run(config: CollectorConfig, args: argparse.Namespace) -> int:
    return cmd_schedule(config, args)


def cmd_status(config: CollectorConfig, args: argparse.Namespace) -> int:
    from .api import UsageAPI

    ledger = Ledger(_ensure_ledger_path(config))
    try:
        api = UsageAPI(config, ledger, fixture_root=args.fixture_root)
        _print(api.status(args.account))
        return 0
    finally:
        ledger.close()


def cmd_report(config: CollectorConfig, args: argparse.Namespace) -> int:
    ledger = Ledger(_ensure_ledger_path(config))
    try:
        start = _parse_since(getattr(args, "start", None) or getattr(args, "since_from", None))
        if getattr(args, "start", None) is None and getattr(args, "from_time", None):
            start = _parse_since(args.from_time)
        end = parse_datetime(args.end) if getattr(args, "end", None) else None
        if getattr(args, "to_time", None):
            end = parse_datetime(args.to_time)
        lookback = _lookback(getattr(args, "lookback", None) or getattr(args, "since", None))
        if getattr(args, "from_time", None) or getattr(args, "start", None):
            lookback = None
        report = build_report(
            config,
            ledger,
            account_id=args.account,
            start=start if lookback is None else None,
            end=end,
            lookback=lookback,
            window_bucket=getattr(args, "window_bucket", None) or (
                None if getattr(args, "window", None) in (None, "current") else args.window
            ),
        )
        _print(report)
        return 0
    finally:
        ledger.close()


def cmd_quota(config: CollectorConfig, args: argparse.Namespace) -> int:
    ledger = Ledger(_ensure_ledger_path(config))
    try:
        account = config.account(args.account)
        if getattr(args, "quota_command", None) == "record-observation" or getattr(args, "record_observation", False):
            observed_at = parse_datetime(args.as_of) if getattr(args, "as_of", None) else datetime.now(timezone.utc)
            recorded = record_manual_observation(
                ledger,
                account_id=account.id,
                bucket_id=args.bucket,
                remaining=None if args.remaining is None else int(args.remaining),
                capacity=None if getattr(args, "capacity", None) is None else int(args.capacity),
                observed_at=ensure_utc(observed_at or datetime.now(timezone.utc)),
                source=getattr(args, "source", None) or "operator-ui",
            )
            _print({"status": "recorded", "observation": recorded, "window_unchanged": True})
            return 0
        from .api import UsageAPI

        api = UsageAPI(config, ledger)
        _print(api.windows(account.id))
        return 0
    finally:
        ledger.close()


def cmd_windows(config: CollectorConfig, args: argparse.Namespace) -> int:
    ledger = Ledger(_ensure_ledger_path(config))
    try:
        account = config.account(args.account)
        action = getattr(args, "windows_command", None)
        if action == "set-explicit":
            start = parse_datetime(args.start)
            if start is None:
                print("windows set-explicit requires --start", file=sys.stderr)
                return 2
            stored = set_explicit_window(
                ledger,
                account_id=account.id,
                bucket_id=args.bucket,
                start=start,
                end=parse_datetime(args.end),
                evidence=args.evidence,
                reason=args.reason,
            )
            _print({"status": "set", "window": stored})
            return 0
        from .api import UsageAPI

        _print(UsageAPI(config, ledger).windows(account.id))
        return 0
    finally:
        ledger.close()


def cmd_schedule(config: CollectorConfig, args: argparse.Namespace) -> int:
    ledger = Ledger(_ensure_ledger_path(config))
    try:
        collector = Collector(config, ledger, fixture_root=args.fixture_root)
        scheduler = Scheduler(config, ledger, collector)
        account = config.account(args.account)
        if getattr(args, "schedule_command", None) == "set" or getattr(args, "set_interval", None) or getattr(args, "every", None):
            raw = getattr(args, "every", None) or getattr(args, "set_interval", None)
            if raw is None:
                print("schedule set requires --every", file=sys.stderr)
                return 2
            raw_s = str(raw)
            if raw_s.startswith("P"):
                interval = parse_iso_duration(raw_s)
            elif raw_s.lower().endswith("h") and raw_s[:-1].isdigit():
                interval = parse_iso_duration(f"PT{int(raw_s[:-1])}H")
            else:
                interval = parse_iso_duration(raw_s)
            _print(scheduler.set_interval(account, interval))
            return 0
        loop = args.command == "run" and not getattr(args, "once", False)
        if not loop:
            result = scheduler.run_if_due(
                account_id=args.account,
                force=getattr(args, "force", False),
                mode=getattr(args, "mode", None) or "refresh",
            )
            if result is None:
                _print({"status": "not_due", "result": "not_due"})
                return 0
            _print(
                {
                    "run_id": result.run_id,
                    "mode": result.mode,
                    "result": result.result,
                    "new_attempts": result.new_attempts,
                    "missed_intervals": result.missed_intervals,
                }
            )
            return 0 if result.result == "complete" else 1
        print(f"Scheduler starting for account {account.id} (Ctrl+C to stop)", file=sys.stderr)
        try:
            while True:
                result = scheduler.run_if_due(account_id=account.id, force=False, mode="refresh")
                if result is not None:
                    _print(
                        {
                            "ts": isoformat_utc(datetime.now(timezone.utc)),
                            "run_id": result.run_id,
                            "result": result.result,
                            "new_attempts": result.new_attempts,
                            "missed_intervals": result.missed_intervals,
                        }
                    )
                import time

                time.sleep(60)
        except KeyboardInterrupt:
            print("Scheduler stopped.", file=sys.stderr)
        return 0
    finally:
        ledger.close()


def cmd_rebuild(config: CollectorConfig, args: argparse.Namespace) -> int:
    ledger = Ledger(_ensure_ledger_path(config))
    try:
        apply = bool(getattr(args, "apply", False)) and not bool(getattr(args, "dry_run", False))
        if getattr(args, "dry_run", False):
            apply = False
        result = rebuild_aggregates(config, ledger, account_id=args.account, apply=apply)
        _print(result)
        return 0
    finally:
        ledger.close()


def cmd_retention(config: CollectorConfig, args: argparse.Namespace) -> int:
    ledger = Ledger(_ensure_ledger_path(config))
    try:
        account = config.account(args.account)
        now = parse_datetime(getattr(args, "as_of", None) or "") or datetime.now(timezone.utc)
        result = ledger.prune_retention(
            account.id,
            now=ensure_utc(now),
            observation_days=config.retention.sanitized_observation_days,
            attempt_days=config.retention.normalized_attempt_days,
            daily_aggregate_days=config.retention.daily_aggregate_days,
            preserve_active_window_evidence=config.retention.preserve_active_window_evidence,
        )
        _print(result)
        return 0
    finally:
        ledger.close()


def cmd_export(config: CollectorConfig, args: argparse.Namespace) -> int:
    ledger = Ledger(_ensure_ledger_path(config))
    try:
        lookback = _lookback(getattr(args, "since", None) or "7d")
        report = build_report(config, ledger, account_id=args.account, lookback=lookback)
        fmt = getattr(args, "format", None) or "json"
        if fmt == "json":
            _print(report)
            return 0
        if fmt == "markdown":
            print("# ChatGPT Chat usage export")
            print()
            print("Relevance: 100% to requested interval")
            print("Confidence: B for fixture-backed reconstructed counts; C for quota estimates.")
            print()
            print(json.dumps(report, indent=2, default=str))
            return 0
        print("account_id,bucket_id,working_usage_estimate,working_remaining_estimate", file=sys.stdout)
        for bucket in report.get("quota_buckets") or []:
            print(
                f"{report['account_id']},{bucket['bucket_id']},{bucket.get('working_usage_estimate')},{bucket.get('working_remaining_estimate')}"
            )
        return 0
    finally:
        ledger.close()


def cmd_dashboard(config: CollectorConfig, args: argparse.Namespace) -> int:
    from .api import serve

    ledger = Ledger(_ensure_ledger_path(config))
    return serve(config, ledger, fixture_root=args.fixture_root)


def cmd_models_review(config: CollectorConfig, args: argparse.Namespace) -> int:
    _print(
        {
            "mapping_version": config.model_mapping.version,
            "canonical_families": list(config.model_mapping.canonical_families),
            "exact_rules": list(config.model_mapping.exact_rules),
            "unknown_behavior": config.model_mapping.unknown_behavior,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.chatgpt_chat_usage_capture",
        description="ChatGPT ordinary Chat usage collector (D1-752). Lookback reports are elapsed time; quota windows are explicit.",
    )
    parser.add_argument("--config", default="scripts/chatgpt_chat_usage_capture/config.yaml", help="Path to YAML config file")
    parser.add_argument("--account", default=None, help="Account ID")
    parser.add_argument("--fixture-root", default=None, help="Fixture directory for adapter")
    sub = parser.add_subparsers(dest="command", help="Subcommands")

    sub.add_parser("init", help="Create local ledger paths and account rows")
    sub.add_parser("browser-login", help="Describe interactive browser login (no credentials stored here)")
    sub.add_parser("inspect", help="Show adapter capabilities and identity")
    sub.add_parser("inspect-capabilities", help="Show adapter capabilities and identity")
    models = sub.add_parser("models", help="Model mapping review")
    models_sub = models.add_subparsers(dest="models_command")
    models_sub.add_parser("review", help="Show canonical families and mapping version")

    collect = sub.add_parser("collect", help="Run a single collection")
    collect.add_argument("--mode", default=None, choices=["refresh", "backfill", "reconcile"])
    collect.add_argument("--since", default=None, help="ISO-8601 datetime or 14d/24h lookback")

    backfill = sub.add_parser("backfill", help="Force a backfill collection")
    backfill.add_argument("--since", default=None, help="ISO-8601 datetime or 14d lookback")

    sub.add_parser("refresh", help="Force a refresh collection using the shared lease")
    run = sub.add_parser("run", help="Run the scheduler loop")
    run.add_argument("--once", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--mode", default=None, choices=["refresh", "backfill", "reconcile"])

    sub.add_parser("status", help="Show freshness, next due, and quota cards")

    report = sub.add_parser("report", help="Show usage report. --since is elapsed lookback; --window is a quota window.")
    report.add_argument("--start", default=None, help="ISO-8601 start datetime")
    report.add_argument("--end", default=None, help="ISO-8601 end datetime")
    report.add_argument("--from", dest="from_time", default=None, help="ISO-8601 start datetime")
    report.add_argument("--to", dest="to_time", default=None, help="ISO-8601 end datetime")
    report.add_argument("--lookback", default=None, help="ISO-8601 duration lookback")
    report.add_argument("--since", default=None, help="Elapsed lookback such as 7d or 24h")
    report.add_argument("--window-bucket", default=None, help="Bucket ID for window-based report")
    report.add_argument("--window", default=None, help="current or bucket id")
    report.add_argument("--group-by", default=None)
    report.add_argument("--format", default="json")

    quota = sub.add_parser("quota", help="Show quota windows or record a snapshot without setting a window")
    quota.add_argument("--set-window", default=None, help="legacy BUCKET_ID,TYPE[,START,END]")
    quota.add_argument("--reset-window", default=None, nargs="?", const="__all__")
    quota_sub = quota.add_subparsers(dest="quota_command")
    record = quota_sub.add_parser("record-observation", help="Record remaining shown by the website")
    record.add_argument("--bucket", required=True)
    record.add_argument("--remaining", required=True)
    record.add_argument("--as-of", dest="as_of", default=None)
    record.add_argument("--source", default="operator-ui")
    record.add_argument("--capacity", default=None)

    windows = sub.add_parser("windows", help="List or set explicit quota windows")
    windows_sub = windows.add_subparsers(dest="windows_command")
    windows_sub.add_parser("list", help="List window definitions")
    set_explicit = windows_sub.add_parser("set-explicit", help="Set operator-explicit bounds")
    set_explicit.add_argument("--bucket", required=True)
    set_explicit.add_argument("--start", required=True)
    set_explicit.add_argument("--end", default=None)
    set_explicit.add_argument("--evidence", required=True)
    set_explicit.add_argument("--reason", required=True)

    sched = sub.add_parser("schedule", help="Run scheduler or change interval")
    sched.add_argument("--once", action="store_true")
    sched.add_argument("--force", action="store_true")
    sched.add_argument("--mode", default=None, choices=["refresh", "backfill", "reconcile"])
    sched.add_argument("--set-interval", default=None)
    sched_sub = sched.add_subparsers(dest="schedule_command")
    sched_set = sched_sub.add_parser("set", help="Recalculate next due from the UTC anchor")
    sched_set.add_argument("--every", required=True, help="PT3H or 3h")

    rebuild = sub.add_parser("rebuild", help="Rebuild aggregates from the ledger without website I/O")
    rebuild.add_argument("--dry-run", action="store_true")
    rebuild.add_argument("--apply", action="store_true")

    retention = sub.add_parser("retention", help="Apply local retention without website I/O")
    retention_sub = retention.add_subparsers(dest="retention_command")
    prune = retention_sub.add_parser("prune", help="Tombstone expired attempts; keep aliases")
    prune.add_argument("--as-of", dest="as_of", default=None)

    export = sub.add_parser("export", help="Export a lookback report")
    export.add_argument("--since", default="7d")
    export.add_argument("--format", default="json", choices=["json", "csv", "markdown"])

    sub.add_parser("dashboard", help="Start local HTTP dashboard and /api/v1")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2
    command = args.command
    if command == "inspect-capabilities":
        command = "inspect"
    if command == "models":
        return cmd_models_review(config, args)
    cmd_map = {
        "init": cmd_init,
        "browser-login": cmd_browser_login,
        "inspect": cmd_inspect,
        "collect": cmd_collect,
        "backfill": cmd_backfill,
        "refresh": cmd_refresh,
        "run": cmd_run,
        "status": cmd_status,
        "report": cmd_report,
        "quota": cmd_quota,
        "windows": cmd_windows,
        "schedule": cmd_schedule,
        "rebuild": cmd_rebuild,
        "retention": cmd_retention,
        "export": cmd_export,
        "dashboard": cmd_dashboard,
    }
    handler = cmd_map.get(command)
    if handler is None:
        print(f"unknown command: {args.command}", file=sys.stderr)
        return 2
    try:
        return handler(config, args)
    except (ConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
