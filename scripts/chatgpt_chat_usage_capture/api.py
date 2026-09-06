"""Loopback HTTP API and dashboard for ChatGPT Chat usage capture."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from .accounting import (
    record_manual_observation,
    rebuild_aggregates,
    set_explicit_window,
)
from .collector import Collector
from .config import CollectorConfig
from .ledger import Ledger
from .privacy import SURFACE_CHAT
from .reporting import build_report
from .scheduler import LeaseHeldError, Scheduler
from .timeutil import ensure_utc, isoformat_utc, parse_datetime, parse_iso_duration


class APIError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise APIError(400, f"invalid json: {exc}") from exc
    if not isinstance(payload, dict):
        raise APIError(400, "json body must be an object")
    return payload


def _query(path: str) -> dict[str, str]:
    parsed = urlparse(path)
    query = parse_qs(parsed.query)
    return {key: values[-1] for key, values in query.items() if values}


def _require_account(config: CollectorConfig, account_id: Optional[str]) -> str:
    return config.account(account_id).id


def _parse_when(value: Optional[str], *, default: Optional[datetime] = None) -> Optional[datetime]:
    if value in (None, ""):
        return default
    parsed = parse_datetime(value)
    if parsed is None:
        raise APIError(400, f"unrecognized timestamp: {value}")
    return parsed


class UsageAPI:
    def __init__(
        self,
        config: CollectorConfig,
        ledger: Ledger,
        *,
        collector: Collector | None = None,
        scheduler: Scheduler | None = None,
        fixture_root: Optional[str] = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.collector = collector or Collector(config, ledger, fixture_root=fixture_root)
        self.scheduler = scheduler or Scheduler(config, ledger, self.collector)

    def authorized(self, headers: Any) -> bool:
        if self.config.application.local_api_auth != "required":
            return True
        token = self.config.application.local_api_token or "local-dev"
        supplied = headers.get("Authorization") or headers.get("X-API-Token") or ""
        if supplied == f"Bearer {token}" or supplied == token:
            return True
        return False

    def csrf_ok(self, method: str, headers: Any, origin_required: bool = False) -> bool:
        if method in {"GET", "HEAD"}:
            return True
        origin = headers.get("Origin") or headers.get("Referer") or ""
        host = headers.get("Host") or f"{self.config.application.bind_host}:{self.config.application.bind_port}"
        if origin and not (
            origin.startswith("http://127.0.0.1")
            or origin.startswith("http://localhost")
            or origin.startswith(f"http://{host}")
        ):
            return False
        if origin_required and not origin:
            return False
        return True

    def dispatch(self, method: str, path: str, headers: Any, body: dict[str, Any]) -> tuple[int, dict[str, Any] | str]:
        parsed = urlparse(path)
        route = parsed.path.rstrip("/") or "/"
        query = _query(path)
        if method == "GET" and route == "/health/live":
            return 200, {"status": "live", "surface": SURFACE_CHAT}
        if method == "GET" and route == "/health/ready":
            return 200, self.status(query.get("account"))
        if method == "GET" and route == "/health":
            return 200, {"status": "ok", "surface": SURFACE_CHAT}
        if method == "GET" and route == "/api/v1/accounts":
            return 200, {"accounts": self.accounts()}
        if method == "GET" and route == "/api/v1/status":
            return 200, self.status(query.get("account"))
        if method == "GET" and route == "/api/v1/usage":
            return 200, self.usage(query)
        if method == "GET" and route == "/api/v1/attempts":
            return 200, self.attempts(query)
        if method == "GET" and route == "/api/v1/quota-windows":
            return 200, {"windows": self.windows(query.get("account"))}
        if method == "GET" and route == "/api/v1/quota-observations":
            return 200, {"observations": self.observations(query.get("account"), query.get("bucket"))}
        if method == "GET" and route == "/api/v1/coverage":
            return 200, self.coverage(query)
        if method == "GET" and route == "/api/v1/runs":
            return 200, {"runs": self.ledger.list_runs(_require_account(self.config, query.get("account")))}
        if method == "POST" and route == "/api/v1/refresh":
            return 200, self.refresh(body)
        if method == "POST" and route == "/api/v1/schedule":
            return 200, self.schedule(body)
        if method == "POST" and route == "/api/v1/window-definitions":
            return 200, self.set_window(body)
        if method == "POST" and route == "/api/v1/manual-observations":
            return 200, self.manual_observation(body)
        if method == "POST" and route == "/api/v1/rebuild-preview":
            return 200, rebuild_aggregates(self.config, self.ledger, account_id=body.get("account"), apply=False)
        if method == "POST" and route == "/api/v1/rebuild-apply":
            return 200, rebuild_aggregates(self.config, self.ledger, account_id=body.get("account"), apply=True)
        # Compatibility aliases used by the seed dashboard.
        if method == "GET" and route == "/api/report":
            return 200, self.usage(query)
        if method == "GET" and route == "/api/attempts":
            return 200, self.attempts(query)
        if method == "GET" and route == "/api/quota":
            return 200, {"buckets": self.usage(query).get("quota_buckets") or []}
        raise APIError(404, f"not found: {route}")

    def accounts(self) -> list[dict[str, Any]]:
        rows = {item["collector_account_id"]: item for item in self.ledger.list_accounts()}
        result = []
        for account in self.config.accounts:
            stored = rows.get(account.id) or {}
            result.append(
                {
                    "id": account.id,
                    "enabled": account.enabled,
                    "quota_owner_id": account.quota_owner_id,
                    "surface": SURFACE_CHAT,
                    "plan_policy_id": account.plan_policy_id,
                    "auth_state": stored.get("auth_state") or "unconfigured",
                }
            )
        return result

    def status(self, account_id: Optional[str]) -> dict[str, Any]:
        account = self.config.account(account_id)
        report = build_report(self.config, self.ledger, account_id=account.id)
        state = self.ledger.get_scheduler_state(account.id) or {}
        return {
            "account_id": account.id,
            "surface": SURFACE_CHAT,
            "freshness": report["freshness"],
            "next_due_at": state.get("next_due_at"),
            "backoff_until": state.get("backoff_until"),
            "lease_owner": state.get("lease_owner"),
            "missed_intervals": int(state.get("missed_intervals") or 0),
            "quota_buckets": report["quota_buckets"],
            "label": report["label"],
        }

    def usage(self, query: dict[str, str]) -> dict[str, Any]:
        account_id = query.get("account")
        start = _parse_when(query.get("from") or query.get("start"))
        end = _parse_when(query.get("to") or query.get("end"))
        lookback = parse_iso_duration(query["lookback"]) if query.get("lookback") else None
        return build_report(
            self.config,
            self.ledger,
            account_id=account_id,
            start=start,
            end=end,
            lookback=lookback,
            window_bucket=query.get("window") or query.get("window_bucket"),
        )

    def attempts(self, query: dict[str, str]) -> dict[str, Any]:
        account_id = _require_account(self.config, query.get("account"))
        cursor = query.get("cursor")
        limit = int(query.get("limit") or 100)
        rows = self.ledger.list_attempts(account_id, cursor=cursor, limit=limit + 1)
        next_cursor = None
        if len(rows) > limit:
            next_cursor = rows[limit]["attempt_id"]
            rows = rows[:limit]
        return {"attempts": rows, "count": len(rows), "cursor": next_cursor}

    def windows(self, account_id: Optional[str]) -> list[dict[str, Any]]:
        account = self.config.account(account_id)
        policy = self.config.policy_for(account)
        stored = {item["bucket_id"]: item for item in self.ledger.list_windows(account.id)}
        rows = []
        for bucket in policy.buckets:
            item = stored.get(bucket.id) or {}
            rows.append(
                {
                    "bucket_id": bucket.id,
                    "families": list(bucket.families),
                    "capacity": bucket.capacity,
                    "window_type": item.get("window_type") or bucket.window.type,
                    "start_at": item.get("start_at") or isoformat_utc(bucket.window.start),
                    "end_at": item.get("end_at") or isoformat_utc(bucket.window.end),
                    "evidence": item.get("evidence") or bucket.window.evidence,
                    "reason": item.get("reason") or bucket.window.reason,
                }
            )
        return rows

    def observations(self, account_id: Optional[str], bucket_id: Optional[str]) -> list[dict[str, Any]]:
        account = self.config.account(account_id)
        return self.ledger.list_quota_observations(account.id, bucket_id=bucket_id)

    def coverage(self, query: dict[str, str]) -> dict[str, Any]:
        account = self.config.account(query.get("account"))
        return {
            "account_id": account.id,
            "gaps": self.ledger.list_coverage_gaps(account.id),
            "runs": self.ledger.list_runs(account.id, limit=10),
            "range": {"from": query.get("from"), "to": query.get("to")},
        }

    def refresh(self, body: dict[str, Any]) -> dict[str, Any]:
        account_id = body.get("account")
        mode = str(body.get("mode") or "refresh")
        force = bool(body.get("force"))
        since = _parse_when(body.get("since"))
        try:
            result = self.scheduler.run_if_due(account_id, force=force or mode in {"backfill", "reconcile"}, mode=mode, since=since)
        except LeaseHeldError as exc:
            return {"status": "coalesced", "reason": str(exc)}
        if result is None:
            return {"status": "not_due"}
        return {
            "status": result.result,
            "run_id": result.run_id,
            "mode": result.mode,
            "missed_intervals": result.missed_intervals,
            "new_attempts": result.new_attempts,
        }

    def schedule(self, body: dict[str, Any]) -> dict[str, Any]:
        account = self.config.account(body.get("account"))
        every = body.get("every") or body.get("refresh_interval")
        if not every:
            raise APIError(400, "every/refresh_interval is required")
        interval = parse_iso_duration(str(every)) if str(every).startswith("P") else parse_iso_duration(_hours_to_iso(str(every)))
        return self.scheduler.set_interval(account, interval)

    def set_window(self, body: dict[str, Any]) -> dict[str, Any]:
        account = self.config.account(body.get("account"))
        start = _parse_when(body.get("start"))
        if start is None:
            raise APIError(400, "start is required")
        stored = set_explicit_window(
            self.ledger,
            account_id=account.id,
            bucket_id=str(body.get("bucket") or body.get("bucket_id") or ""),
            start=start,
            end=_parse_when(body.get("end")),
            evidence=str(body.get("evidence") or ""),
            reason=str(body.get("reason") or ""),
            window_type=str(body.get("window_type") or "operator_explicit"),
            timezone_name=body.get("timezone"),
            duration=body.get("duration"),
        )
        return {"status": "set", "window": stored}

    def manual_observation(self, body: dict[str, Any]) -> dict[str, Any]:
        account = self.config.account(body.get("account"))
        observed_at = _parse_when(body.get("as_of") or body.get("observed_at"), default=datetime.now(timezone.utc))
        remaining = body.get("remaining")
        capacity = body.get("capacity")
        recorded = record_manual_observation(
            self.ledger,
            account_id=account.id,
            bucket_id=str(body.get("bucket") or body.get("bucket_id") or ""),
            remaining=None if remaining is None else int(remaining),
            capacity=None if capacity is None else int(capacity),
            observed_at=ensure_utc(observed_at or datetime.now(timezone.utc)),
            source=str(body.get("source") or "operator-ui"),
            reset_at=_parse_when(body.get("reset_at")),
            notes=body.get("notes"),
        )
        return {"status": "recorded", "observation": recorded, "window_unchanged": True}


def _hours_to_iso(raw: str) -> str:
    value = raw.strip().lower()
    if value.endswith("h") and value[:-1].isdigit():
        return f"PT{int(value[:-1])}H"
    if value.endswith("m") and value[:-1].isdigit():
        return f"PT{int(value[:-1])}M"
    raise APIError(400, f"unrecognized interval: {raw}")


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>ChatGPT Chat Usage</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;margin:2rem;background:#fafafa;color:#222}
table{border-collapse:collapse;width:100%;max-width:1100px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #ddd}
th{font-weight:600;background:#f0f0f0}
.cards{display:flex;gap:1rem;flex-wrap:wrap}
.bucket{background:#fff;border:1px solid #ddd;padding:1rem;margin:1rem 0;border-radius:4px;min-width:240px}
.unknown{color:#8a5a00;font-weight:600}
</style></head>
<body>
<h1>ChatGPT Chat Usage</h1>
<p>Surface: <strong>chat</strong> (ordinary Chat only). Remaining values are working estimates, not official quota.</p>
<p id="freshness">Loading freshness...</p>
<div class="cards" id="quota"></div>
<div id="report"></div>
<script>
function fmt(value){return (value===null||value===undefined)?"Unknown":value;}
const TOKEN = window.__USAGE_TOKEN || "";
const headers = TOKEN ? {"Authorization": "Bearer " + TOKEN} : {};
async function load(){
  const status=await (await fetch("/api/v1/status", {headers})).json();
  const usage=await (await fetch("/api/v1/usage", {headers})).json();
  const f=status.freshness||{};
  document.getElementById("freshness").textContent=
    "Freshness: "+fmt(f.status)+" | next due: "+fmt(f.next_due_at)+" | last completed: "+fmt(f.history_last_completed_at);
  let cards="";
  for(const item of (status.quota_buckets||[])){
    const remaining=item.working_remaining_estimate;
    cards+="<div class=bucket><strong>"+item.bucket_id+"</strong>"
      +"<div>capacity: "+fmt(item.capacity)+"</div>"
      +"<div>window: "+fmt(item.window&&item.window.type)+"</div>"
      +"<div>used: "+fmt(item.working_usage_estimate)+"</div>"
      +"<div>remaining: <span class="+(remaining===null?"unknown":"")+">"+fmt(remaining)+"</span></div>"
      +"<div>server remaining: "+fmt(item.server_reported_remaining)+"</div>"
      +"<div>as of: "+fmt(item.evaluated_at)+"</div></div>";
  }
  document.getElementById("quota").innerHTML=cards||"<p>No quota buckets configured.</p>";
  let days="<h2>Last 7 calendar days (display timezone)</h2><table><tr><th>Date</th><th>Requested</th><th>Completed</th></tr>";
  for(const row of (usage.calendar_days||[])){
    days+="<tr><td>"+row.date+"</td><td>"+JSON.stringify(row.observed_attempts_by_requested_family||{})+"</td><td>"+JSON.stringify(row.completed_answers_by_recorded_final_family||{})+"</td></tr>";
  }
  days+="</table>";
  document.getElementById("report").innerHTML=days;
}
load();
</script>
</body>
</html>
"""


def make_handler(api: UsageAPI):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self._handle("GET")

        def do_POST(self):  # noqa: N802
            self._handle("POST")

        def _handle(self, method: str) -> None:
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"
            if method == "GET" and route in {"/", "/index.html"}:
                token = api.config.application.local_api_token or "local-dev"
                body = DASHBOARD_HTML.replace(
                    "<script>",
                    "<script>window.__USAGE_TOKEN=" + json.dumps(token) + ";",
                    1,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if not api.authorized(self.headers):
                self._write(401, {"error": "unauthorized"})
                return
            if not api.csrf_ok(method, self.headers):
                self._write(403, {"error": "csrf_rejected"})
                return
            try:
                body = _json_body(self) if method == "POST" else {}
                status, payload = api.dispatch(method, self.path, self.headers, body)
            except APIError as exc:
                self._write(exc.status, {"error": exc.message})
                return
            except Exception as exc:  # pragma: no cover - defensive
                self._write(400, {"error": str(exc)})
                return
            self._write(status, payload)

        def _write(self, status: int, payload: dict[str, Any] | str) -> None:
            body = payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str)
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def serve(config: CollectorConfig, ledger: Ledger, *, fixture_root: Optional[str] = None) -> int:
    api = UsageAPI(config, ledger, fixture_root=fixture_root)
    handler = make_handler(api)
    host = config.application.bind_host
    port = config.application.bind_port
    server = HTTPServer((host, port), handler)
    print(f"Dashboard at http://{host}:{port}", file=__import__("sys").stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        ledger.close()
    return 0
