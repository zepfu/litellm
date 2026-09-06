"""ChatGPT ordinary Chat usage collector CLI (D1-752)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .collector import Collector
from .config import CollectorConfig, ConfigError, load_config
from .ledger import Ledger
from .privacy import SURFACE_CHAT
from .reporting import build_report
from .scheduler import Scheduler
from .timeutil import ensure_utc, isoformat_utc, parse_iso_duration


def _ensure_ledger_path(config: CollectorConfig) -> Path:
    path = config.application.database_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cmd_inspect(config: CollectorConfig, args: argparse.Namespace) -> int:
    """Print adapter capabilities and identity for the configured account."""
    ledger = Ledger(_ensure_ledger_path(config))
    try:
        collector = Collector(config, ledger, fixture_root=args.fixture_root)
        result = collector.inspect_capabilities(args.account)
        print(json.dumps(result, indent=2, default=str))
        return 0
    finally:
        ledger.close()


def cmd_collect(config: CollectorConfig, args: argparse.Namespace) -> int:
    """Run a single collection cycle."""
    ledger = Ledger(_ensure_ledger_path(config))
    try:
        collector = Collector(config, ledger, fixture_root=args.fixture_root)
        since = None
        if args.since:
            since = ensure_utc(datetime.fromisoformat(args.since))
        result = collector.collect(
            account_id=args.account,
            mode=args.mode or "refresh",
            since=since,
        )
        print(json.dumps(
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
            },
            indent=2,
            default=str,
        ))
        return 0 if result.result == "complete" else 1
    finally:
        ledger.close()


def cmd_report(config: CollectorConfig, args: argparse.Namespace) -> int:
    """Print a Chat usage report."""
    ledger = Ledger(_ensure_ledger_path(config))
    try:
        start = None
        end = None
        if args.start:
            start = ensure_utc(datetime.fromisoformat(args.start))
        if args.end:
            end = ensure_utc(datetime.fromisoformat(args.end))
        lookback = None
        if args.lookback:
            lookback = parse_iso_duration(args.lookback)
        report = build_report(
            config,
            ledger,
            account_id=args.account,
            start=start,
            end=end,
            lookback=lookback,
            window_bucket=args.window_bucket,
        )
        print(json.dumps(report, indent=2, default=str))
        return 0
    finally:
        ledger.close()


def cmd_quota(config: CollectorConfig, args: argparse.Namespace) -> int:
    """Show or set quota window definitions."""
    ledger = Ledger(_ensure_ledger_path(config))
    try:
        account = config.account(args.account)
        policy = config.policy_for(account)
        if args.set_window:
            parts = args.set_window.split(",", 3)
            if len(parts) < 2:
                print("usage: --set-window BUCKET_ID,TYPE[,START,END]", file=sys.stderr)
                return 2
            bucket_id = parts[0].strip()
            window_type = parts[1].strip()
            start = ensure_utc(datetime.fromisoformat(parts[2])) if len(parts) > 2 and parts[2] else None
            end = ensure_utc(datetime.fromisoformat(parts[3])) if len(parts) > 3 and parts[3] else None
            ledger.set_window(
                account.id,
                bucket_id,
                window_type=window_type,
                start=start,
                end=end,
                evidence="cli_override",
                reason="operator_set",
            )
            print(json.dumps({"status": "set", "bucket_id": bucket_id, "window_type": window_type}))
        elif args.reset_window:
            now = ensure_utc(datetime.now(timezone.utc))
            for bucket in policy.buckets:
                if bucket.id == args.reset_window or not args.reset_window:
                    ledger.set_window(
                        account.id,
                        bucket.id,
                        window_type="operator_explicit",
                        start=now,
                        end=None,
                        evidence="cli_reset",
                        reason="operator_reset",
                    )
        else:
            lines = []
            for bucket in policy.buckets:
                stored = ledger.get_window(account.id, bucket.id)
                server = ledger.latest_quota_observation(account.id, bucket.id)
                lines.append({
                    "bucket_id": bucket.id,
                    "families": list(bucket.families),
                    "capacity": bucket.capacity,
                    "window_type": stored["window_type"] if stored else bucket.window.type,
                    "window_start": stored["start_at"] if stored else isoformat_utc(bucket.window.start),
                    "window_end": stored["end_at"] if stored else isoformat_utc(bucket.window.end),
                    "server_remaining": server["remaining"] if server else None,
                    "server_observed_at": server["observed_at"] if server else None,
                })
            print(json.dumps(lines, indent=2, default=str))
        return 0
    finally:
        ledger.close()


def cmd_schedule(config: CollectorConfig, args: argparse.Namespace) -> int:
    """Run the scheduler loop (or one-shot if --once)."""
    ledger = Ledger(_ensure_ledger_path(config))
    try:
        collector = Collector(config, ledger, fixture_root=args.fixture_root)
        scheduler = Scheduler(config, ledger, collector)
        account = config.account(args.account)
        if args.set_interval:
            interval = parse_iso_duration(args.set_interval)
            interval_result = scheduler.set_interval(account, interval)
            print(json.dumps(interval_result, indent=2, default=str))
            return 0
        if args.once:
            result = scheduler.run_if_due(
                account_id=args.account,
                force=args.force,
                mode=args.mode or "refresh",
            )
            if result is None:
                print(json.dumps({"status": "not_due"}))
                return 0
            print(json.dumps(
                {
                    "run_id": result.run_id,
                    "mode": result.mode,
                    "result": result.result,
                    "new_attempts": result.new_attempts,
                    "missed_intervals": result.missed_intervals,
                },
                indent=2,
            ))
            return 0 if result.result == "complete" else 1
        print(f"Scheduler starting for account {account.id} (Ctrl+C to stop)", file=sys.stderr)
        try:
            while True:
                result = scheduler.run_if_due(
                    account_id=account.id,
                    force=False,
                    mode="refresh",
                )
                if result is not None:
                    print(json.dumps({
                        "ts": isoformat_utc(datetime.now(timezone.utc)),
                        "run_id": result.run_id,
                        "result": result.result,
                        "new_attempts": result.new_attempts,
                        "missed_intervals": result.missed_intervals,
                    }))
                import time
                time.sleep(60)
        except KeyboardInterrupt:
            print("Scheduler stopped.", file=sys.stderr)
        return 0
    finally:
        ledger.close()


def cmd_dashboard(config: CollectorConfig, args: argparse.Namespace) -> int:
    """Start a minimal local HTTP dashboard."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    ledger = Ledger(_ensure_ledger_path(config))

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._json({"status": "ok", "surface": SURFACE_CHAT, "account": config.account(args.account).id})
                return
            if self.path == "/api/report":
                report = build_report(config, ledger, account_id=args.account)
                self._json(report)
                return
            if self.path == "/api/attempts":
                attempts = ledger.list_attempts(config.account(args.account).id)
                self._json({"attempts": attempts, "count": len(attempts)})
                return
            if self.path == "/api/quota":
                account = config.account(args.account)
                policy = config.policy_for(account)
                buckets = []
                for bucket in policy.buckets:
                    stored = ledger.get_window(account.id, bucket.id)
                    server = ledger.latest_quota_observation(account.id, bucket.id)
                    buckets.append({
                        "bucket_id": bucket.id,
                        "capacity": bucket.capacity,
                        "window_type": stored["window_type"] if stored else bucket.window.type,
                        "server_remaining": server["remaining"] if server else None,
                    })
                self._json({"buckets": buckets})
                return
            if self.path in ("/", "/index.html"):
                self._html()
                return
            self.send_error(404)

        def _json(self, data):
            body = json.dumps(data, indent=2, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self):
            html = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>ChatGPT Chat Usage</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;margin:2rem;background:#fafafa;color:#222}
table{border-collapse:collapse;width:100%;max-width:900px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #ddd}
th{font-weight:600;background:#f0f0f0}
.bucket{background:#fff;border:1px solid #ddd;padding:1rem;margin:1rem 0;border-radius:4px}
</style></head>
<body>
<h1>ChatGPT Chat Usage</h1>
<p>Surface: <strong>chat</strong> (ordinary Chat only)</p>
<div id="report">Loading...</div>
<div id="quota">Loading...</div>
<script>
async function load(){try{let r=await fetch("/api/report");let d=await r.json();
document.getElementById("report").innerHTML=
"<h2>Report</h2><p>Range: "+d.range.label+" ("+d.range.start+" to "+d.range.end+")</p>"+
"<p>Freshness: "+d.freshness.status+" | Attempts: "+(d.observed_attempts_by_requested_family?JSON.stringify(d.observed_attempts_by_requested_family):"0")+
" | Unclassified: "+d.unclassified_or_ambiguous_attempts+"</p>"}catch(e){document.getElementById("report").textContent="Error: "+e}
try{let q=await fetch("/api/quota");let b=await q.json();let h="<h2>Quota</h2>";
for(let item of b.buckets){h+="<div class=bucket><strong>"+item.bucket_id+"</strong> capacity="+item.capacity+" server_remaining="+item.server_remaining+"</div>"}
document.getElementById("quota").innerHTML=h}catch(e){document.getElementById("quota").textContent="Error: "+e}}
load();
</script>
</body>
</html>"""
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    host = config.application.bind_host
    port = config.application.bind_port
    server = HTTPServer((host, port), DashboardHandler)
    print(f"Dashboard at http://{host}:{port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        ledger.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.chatgpt_chat_usage_capture",
        description="ChatGPT ordinary Chat usage collector (D1-752)",
    )
    parser.add_argument("--config", default="scripts/chatgpt_chat_usage_capture/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--account", default=None, help="Account ID")
    parser.add_argument("--fixture-root", default=None, help="Fixture directory for adapter")
    sub = parser.add_subparsers(dest="command", help="Subcommands")

    sub.add_parser("inspect", help="Show adapter capabilities and identity")

    collect = sub.add_parser("collect", help="Run a single collection")
    collect.add_argument("--mode", default=None, choices=["refresh", "backfill", "reconcile"])
    collect.add_argument("--since", default=None, help="ISO-8601 datetime for backfill start")

    report = sub.add_parser("report", help="Show usage report")
    report.add_argument("--start", default=None, help="ISO-8601 start datetime")
    report.add_argument("--end", default=None, help="ISO-8601 end datetime")
    report.add_argument("--lookback", default=None, help="ISO-8601 duration lookback")
    report.add_argument("--window-bucket", default=None, help="Bucket ID for window-based report")

    quota = sub.add_parser("quota", help="Show or set quota window definitions")
    quota.add_argument("--set-window", default=None, help="BUCKET_ID,TYPE[,START,END]")
    quota.add_argument("--reset-window", default=None, nargs="?", const="__all__",
                       help="Reset window for bucket (or all)")

    sched = sub.add_parser("schedule", help="Run scheduler")
    sched.add_argument("--once", action="store_true", help="Single run instead of loop")
    sched.add_argument("--force", action="store_true", help="Force run even if not due")
    sched.add_argument("--mode", default=None, choices=["refresh", "backfill", "reconcile"])
    sched.add_argument("--set-interval", default=None, help="ISO-8601 duration to change interval")

    sub.add_parser("dashboard", help="Start local HTTP dashboard")
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
    cmd_map = {
        "inspect": cmd_inspect,
        "collect": cmd_collect,
        "report": cmd_report,
        "quota": cmd_quota,
        "schedule": cmd_schedule,
        "dashboard": cmd_dashboard,
    }
    handler = cmd_map.get(args.command)
    if handler is None:
        print(f"unknown command: {args.command}", file=sys.stderr)
        return 2
    return handler(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
