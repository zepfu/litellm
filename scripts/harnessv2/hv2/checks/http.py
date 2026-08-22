"""HTTP health, catalog, and custom-endpoint suite. Paths live in YAML."""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hv2.docker_guard import assert_url_allowed
from hv2.errors import HarnessError
from hv2.identity import merge_request_headers
from hv2.load_config import as_str_list, config_timeouts
from hv2.plan import compiled_aliases


def _http_timeout(config: Mapping[str, Any]) -> int:
    return config_timeouts(config)["http_seconds"]


def request_json(
    config: Mapping[str, Any],
    *,
    base_url: str,
    method: str,
    path: str,
    body: Any | None = None,
    headers: Mapping[str, str] | None = None,
    identity: bool = True,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    assert_url_allowed(url, config)
    data = None
    caller = {"Accept": "application/json", **(headers or {})}
    req_headers = (
        merge_request_headers(config, caller) if identity else {**caller}
    )
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    request = Request(url, data=data, method=method.upper(), headers=req_headers)
    try:
        with urlopen(request, timeout=_http_timeout(config)) as response:
            raw = response.read()
            status = int(getattr(response, "status", 200) or 200)
            text = raw.decode("utf-8", errors="replace")
    except HTTPError as exc:
        raw = exc.read() if exc.fp is not None else b""
        text = raw.decode("utf-8", errors="replace")
        status = int(exc.code)
    except URLError as exc:
        raise HarnessError(f"HTTP {method} {url} failed: {exc}") from exc
    parsed: Any = None
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
    return {"status": status, "text": text, "json": parsed, "url": url}


def check_health(config: Mapping[str, Any], base_url: str) -> dict[str, Any]:
    checks = config.get("checks") if isinstance(config.get("checks"), dict) else {}
    health = checks.get("health") if isinstance(checks.get("health"), dict) else {}
    path = str(health.get("path") or "/health/liveliness")
    expect = [int(item) for item in (health.get("expect_status") or [200])]
    result = request_json(config, base_url=base_url, method="GET", path=path)
    ok = result["status"] in expect
    failures = [] if ok else [f"health {path} returned {result['status']}, expected {expect}"]
    return {**result, "ok": ok, "failures": failures}


def _catalog_ids(payload: Any) -> set[str]:
    ids: set[str] = set()
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict) and row.get("id"):
                    ids.add(str(row["id"]))
        elif isinstance(payload.get("models"), list):
            for row in payload["models"]:
                if isinstance(row, dict) and row.get("id"):
                    ids.add(str(row["id"]))
                elif isinstance(row, str):
                    ids.add(row)
    return ids


def _wants_catalog_assertions(catalog: Any) -> bool:
    if catalog is None or catalog is False:
        return False
    if isinstance(catalog, str):
        return bool(catalog.strip())
    return bool(catalog)


def _catalog_mismatch_failures(
    path: str,
    payload_ids: set[str],
    *,
    required: list[str],
    absent: list[str],
    served: list[str],
) -> list[str]:
    failures: list[str] = []
    missing = [name for name in required if name not in payload_ids]
    if missing:
        failures.append(f"{path} missing compiled aliases: {missing}")
    present_absent = [name for name in absent if name in payload_ids]
    if present_absent:
        failures.append(f"{path} unexpectedly published {present_absent}")
    missing_served = [name for name in served if name not in payload_ids]
    if missing_served:
        failures.append(f"{path} missing served concrete ids: {missing_served}")
    return failures


def check_catalog_http(config: Mapping[str, Any], base_url: str) -> dict[str, Any]:
    checks = config.get("checks") if isinstance(config.get("checks"), dict) else {}
    http = checks.get("http") if isinstance(checks.get("http"), dict) else {}
    suite = http.get("suite") if isinstance(http.get("suite"), list) else []
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in suite:
        if not isinstance(row, dict):
            continue
        if row.get("catalog") != "compiled_aliases":
            continue
        outcome = _eval_http_row(row, config, base_url)
        results.append(outcome)
        failures.extend(outcome.get("failures") or [])
    catalog_failures = _catalog_assertions_from_results(config, results)
    for item in catalog_failures:
        if item not in failures:
            failures.append(item)
    return {"ok": not failures, "failures": failures, "results": results}


def check_http_suite(config: Mapping[str, Any], base_url: str) -> dict[str, Any]:
    checks = config.get("checks") if isinstance(config.get("checks"), dict) else {}
    http = checks.get("http") if isinstance(checks.get("http"), dict) else {}
    suite = http.get("suite") if isinstance(http.get("suite"), list) else []
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in suite:
        if not isinstance(row, dict):
            continue
        outcome = _eval_http_row(row, config, base_url)
        results.append(outcome)
        failures.extend(outcome.get("failures") or [])
    catalog_failures = _catalog_assertions_from_results(config, results)
    for item in catalog_failures:
        if item not in failures:
            failures.append(item)
    return {"ok": not failures, "failures": failures, "results": results}


def _catalog_assertions_from_results(
    config: Mapping[str, Any],
    results: list[dict[str, Any]],
) -> list[str]:
    models = config.get("models") if isinstance(config.get("models"), dict) else {}
    required = compiled_aliases(config)
    absent = as_str_list(models.get("absent_catalog_ids"))
    served = as_str_list(models.get("served_concrete_ids"))
    failures: list[str] = []
    for outcome in results:
        if not _wants_catalog_assertions(outcome.get("catalog")):
            continue
        if outcome.get("status") != 200:
            continue
        path = str(outcome.get("path") or "")
        payload_ids = _catalog_ids(outcome.get("json"))
        failures.extend(
            _catalog_mismatch_failures(
                path,
                payload_ids,
                required=required,
                absent=absent,
                served=served,
            )
        )
    return failures


def _eval_http_row(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    base_url: str,
) -> dict[str, Any]:
    method = str(row.get("method") or "GET").upper()
    path = str(row.get("path") or "/")
    body = row.get("json")
    expect = [int(item) for item in (row.get("expect_status") or [200])]
    reject = [int(item) for item in (row.get("reject_status") or [])]
    result = request_json(
        config,
        base_url=base_url,
        method=method,
        path=path,
        body=body,
        identity=row.get("identity", True) is not False,
    )
    status = int(result["status"])
    failures: list[str] = []
    name = str(row.get("name") or path)
    if expect and status not in expect:
        failures.append(f"{name}: status {status} not in {expect}")
    if status in reject:
        failures.append(f"{name}: status {status} is rejected")
    if row.get("miss_is_pass") and status in expect:
        # Negative: a miss is pass — already encoded as expect_status.
        pass
    return {
        "name": name,
        "method": method,
        "path": path,
        "status": status,
        "ok": not failures,
        "failures": failures,
        "json": result.get("json"),
        "url": result.get("url"),
        "catalog": row.get("catalog"),
    }
