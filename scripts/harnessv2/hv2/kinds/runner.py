"""Walk kind steps from YAML. New step types require Python; lists do not."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Callable

from hv2.artifact import (
    append_jsonl,
    bounded_step_detail,
    durable_jsonl_path,
    git_stamp,
    sha_drift_warning,
    step_is_halt,
    utc_now_iso,
    write_artifact,
)
from hv2.checks.error_jsonl import jsonl_path, scan_new_rows, snapshot_cursor
from hv2.checks.http import check_catalog_http, check_health, check_http_suite
from hv2.checks.logs import scan_log_text
from hv2.checks.orch_evidence import child_spawn_evidence
from hv2.checks.redis_scan import snapshot_redis
from hv2.checks.session_history import session_history_result
from hv2.checks.soft_fail import matching_signatures
from hv2.docker_guard import run_docker
from hv2.drivers import driver_for
from hv2.errors import PlanError
from hv2.instance import inspect_instance
from hv2.load_config import as_str_list
from hv2.pane import _pane_has_any
from hv2.plan import RunPlan


def _record(
    results: list[dict[str, Any]],
    name: str,
    payload: dict[str, Any],
) -> None:
    row = {"name": name, **payload}
    results.append(row)


def run_plan(plan: RunPlan) -> dict[str, Any]:  # noqa: PLR0915
    started = utc_now_iso()
    results: list[dict[str, Any]] = []
    warnings: list[str] = []
    failures: list[str] = []
    resolved = plan.resolved
    if plan.dry_run:
        artifact = {
            "schema": (plan.config.get("artifact") or {}).get("schema"),
            "started": started,
            "finished": utc_now_iso(),
            "dry_run": True,
            "plan": plan.as_dict(),
            "git": git_stamp(),
            "results": [],
            "ok": True,
        }
        if plan.write_artifact is not None:
            write_artifact(plan.write_artifact, artifact, plan.config)
        return artifact

    if resolved is None:
        resolved = inspect_instance(plan.container, plan.config)
        plan = replace(plan, resolved=resolved)

    jsonl = jsonl_path(plan.config)
    jsonl_before = snapshot_cursor(jsonl)
    log_cursor = utc_now_iso()
    git_start = git_stamp()
    durable_path = durable_jsonl_path(
        plan.config,
        started=started,
        kind=plan.kind,
        container=resolved.container,
        commit=str(git_start.get("commit") or ""),
    )
    append_jsonl(
        durable_path,
        {
            "event": "run_start",
            "started": started,
            "kind": plan.kind,
            "tui": plan.tui,
            "instance": {
                "container": resolved.container,
                "base_url": resolved.base_url,
                "host_port": resolved.host_port,
            },
            "git": git_start,
        },
    )
    halted = False

    remaining = list(plan.steps)
    while remaining:
        step = remaining.pop(0)
        step_type = str(step.get("type") or "")
        when = step.get("when")
        if when == "tui_selected" and not plan.tui:
            continue
        if halted:
            payload = {
                "ok": False,
                "skipped": True,
                "reason": "halted_on_logging_regression",
                "failures": [],
            }
            _record(results, step_type, payload)
            append_jsonl(
                durable_path,
                {
                    "event": "step",
                    "name": step_type,
                    "result": "fail",
                    "skipped": True,
                    "reason": "halted_on_logging_regression",
                    "detail": bounded_step_detail(payload),
                },
            )
            continue
        handler = _STEP_HANDLERS.get(step_type)
        if handler is None:
            raise PlanError(f"unknown kind step type {step_type!r}")
        payload = handler(plan, log_cursor=log_cursor, jsonl_before=jsonl_before)
        _record(results, step_type, payload)
        warnings.extend(payload.get("warnings") or [])
        step_failures = list(payload.get("failures") or [])
        failures.extend(step_failures)
        passed = bool(payload.get("ok", False)) and not step_failures
        append_jsonl(
            durable_path,
            {
                "event": "step",
                "name": step_type,
                "result": "pass" if passed else "fail",
                "detail": bounded_step_detail(payload),
            },
        )
        if step_is_halt(payload):
            halted = True

    finished = utc_now_iso()
    git_end = git_stamp()
    drift = sha_drift_warning(git_start, git_end)
    if drift:
        warnings.append(drift)
    artifact = {
        "schema": (plan.config.get("artifact") or {}).get("schema"),
        "started": started,
        "finished": finished,
        "dry_run": False,
        "plan": plan.as_dict(),
        "target": {
            "container": resolved.container,
            "base_url": resolved.base_url,
            "host_port": resolved.host_port,
            "inspect_env": resolved.inspect_env,
        },
        "tui": plan.tui,
        "test": plan.kind,
        "git": git_end,
        "git_start": git_start,
        "git_end": git_end,
        "durable_jsonl": str(durable_path),
        "results": results,
        "warnings": warnings,
        "failures": failures,
        "halted": halted,
        "ok": not failures,
    }
    append_jsonl(
        durable_path,
        {
            "event": "run_end",
            "finished": finished,
            "ok": artifact["ok"],
            "halted": halted,
            "git_start": git_start,
            "git_end": git_end,
            "sha_drift_warning": drift,
            "failures": failures,
            "warnings": warnings,
        },
    )
    if plan.write_artifact is not None:
        write_artifact(plan.write_artifact, artifact, plan.config)
    return artifact


def _read_logs_since(plan: RunPlan, started_at: str) -> str:
    proc = run_docker(
        plan.config,
        ["logs", "--since", started_at or "1s", plan.container],
        container=plan.container,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def _step_health(plan: RunPlan, **_: Any) -> dict[str, Any]:
    assert plan.resolved is not None
    return check_health(plan.config, plan.resolved.base_url)


def _step_http_suite(plan: RunPlan, **_: Any) -> dict[str, Any]:
    assert plan.resolved is not None
    return check_http_suite(plan.config, plan.resolved.base_url)


def _step_catalog_http(plan: RunPlan, **_: Any) -> dict[str, Any]:
    assert plan.resolved is not None
    return check_catalog_http(plan.config, plan.resolved.base_url)


def _step_error_jsonl(plan: RunPlan, *, jsonl_before: int, **_: Any) -> dict[str, Any]:
    return scan_new_rows(plan.config, before_size=jsonl_before)


def _step_redis(plan: RunPlan, **_: Any) -> dict[str, Any]:
    return snapshot_redis(plan.config)


def _plan_models(plan: RunPlan) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in (*plan.models, *plan.orchestration_parents, *plan.orchestration_children):
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _evidence_order(plan: RunPlan) -> list[str]:
    kinds = plan.config.get("kinds") if isinstance(plan.config.get("kinds"), dict) else {}
    spec = kinds.get("orchestration") if isinstance(kinds.get("orchestration"), dict) else {}
    order = spec.get("evidence_order")
    if isinstance(order, list) and order:
        return as_str_list(order)
    checks = plan.config.get("checks") if isinstance(plan.config.get("checks"), dict) else {}
    history = (
        checks.get("session_history")
        if isinstance(checks.get("session_history"), dict)
        else {}
    )
    nested = history.get("evidence_order")
    if isinstance(nested, list) and nested:
        return as_str_list(nested)
    # YAML should supply kinds.orchestration.evidence_order; this is last-resort only.
    return ["session_history_tool_activity", "aawm_child_routes", "tui_transcript"]


def _soft_fail_from_driver(
    plan: RunPlan,
    driver: Any,
    models: list[str],
    extra_text: str = "",
) -> list[dict[str, Any]]:
    parts: list[str] = []
    if extra_text:
        parts.append(extra_text)
    session_up = hasattr(driver, "tmux_has_session") and driver.tmux_has_session()
    if session_up and hasattr(driver, "capture_pane"):
        parts.append(driver.capture_pane() or "")
    text = "\n".join(part for part in parts if part)
    if not text:
        return []
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates: list[str | None] = list(models) if models else [None]
    for model in candidates:
        for row in matching_signatures(plan.config, text=text, model=model):
            name = str(row.get("name") or "")
            if name in seen:
                continue
            seen.add(name)
            hits.append(row)
    return hits


def _with_session_history(plan: RunPlan, payload: dict[str, Any]) -> dict[str, Any]:
    if plan.kind in {"model", "orchestration"} or (plan.kind == "catalog" and plan.tui):
        payload["session_history"] = session_history_result(plan.config)
    return payload


def _step_docker_logs(plan: RunPlan, *, log_cursor: str, **_: Any) -> dict[str, Any]:
    text = _read_logs_since(plan, log_cursor)
    # Catalog listing does not generate route-rollup traffic, but any leftover
    # native uvicorn access line (except health allow_paths) is still a
    # logging-regression halt (scan_log_text leftover_uvicorn).
    require_rollup = plan.kind in {"model", "orchestration"} and bool(plan.tui)
    logs_cfg = (plan.config.get("checks") or {}).get("logs") or {}
    rollup_cfg = logs_cfg.get("rollup") if isinstance(logs_cfg, dict) else {}
    if plan.kind == "platform":
        require_rollup = bool(rollup_cfg.get("required_on_platform"))
    scan = scan_log_text(
        text,
        plan.config,
        attribution_substrings=[plan.container],
        require_rollup=require_rollup,
        plan_models=_plan_models(plan),
        tui=plan.tui,
    )
    scan["bytes"] = len(text)
    return scan


def _ohmypi_select_spec(driver: Any) -> dict[str, Any]:
    spec = getattr(driver, "spec", {}) or {}
    select = spec.get("select_model") if isinstance(spec, dict) else {}
    return select if isinstance(select, dict) else {}


def _step_tui_catalog(plan: RunPlan, **_: Any) -> dict[str, Any]:
    if not plan.tui:
        return {"ok": True, "skipped": True, "failures": []}
    driver = driver_for(plan.tui, plan.config)
    if not hasattr(driver, "catalog_json"):
        raise PlanError(f"TUI {plan.tui!r} cannot run catalog discovery")
    outcome = driver.catalog_json()
    failures = [] if outcome.get("ok") else ["Ohmypi `omp models --json` failed"]
    finds: list[dict[str, Any]] = []
    samples = list(plan.models) or []
    if hasattr(driver, "catalog_find"):
        for model in samples:
            row = driver.catalog_find(model)
            selector = driver.model_selector(model)
            stdout = str(row.get("stdout") or "")
            found = selector in stdout
            if not row.get("ok") or not found:
                failures.append(
                    f"Ohmypi catalog find {model!r} missing selector {selector}"
                )
            finds.append({**row, "selector": selector, "found": found})
        if samples and "litellm-alpha-passthrough" not in str(outcome.get("stdout") or ""):
            failures.append(
                "Ohmypi `omp models --json` missing provider litellm-alpha-passthrough"
            )
    return _with_session_history(
        plan,
        {
            **outcome,
            "finds": finds,
            "failures": failures,
            "ok": not failures,
        },
    )


def _pane_exact_pong(pane: str, prompt: str) -> bool:
    prompt_line = prompt.strip()
    for raw_line in pane.splitlines():
        line = raw_line.strip()
        if line == "PONG" and line != prompt_line:
            return True
    return False


def _step_tui_model(plan: RunPlan, **_: Any) -> dict[str, Any]:  # noqa: PLR0915
    if not plan.tui:
        raise PlanError("model kind requires --tui")
    driver = driver_for(plan.tui, plan.config)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    prompt = str(plan.extra.get("pong_prompt") or "Reply with exactly the word PONG.")
    if not plan.models:
        raise PlanError("model kind has an empty model list")
    select = _ohmypi_select_spec(driver)
    tools = bool(select.get("tools_for_model", False))
    pass_needles = as_str_list(select.get("pass_needles")) or ["PONG"]
    provider_404_needles = (
        as_str_list(select.get("provider_404_needles")) or ["404"]
    )
    reply_needles = as_str_list(select.get("reply_needles")) or (
        pass_needles + provider_404_needles
    )
    send_text = ""
    try:
        for model in plan.models:
            argv = driver.launch_argv(model)
            driver.assert_no_print_flags(argv)
            if "-p" in argv or "--print" in argv:
                failures.append(f"refusing print-mode argv for {model}")
                continue
            launched = driver.ensure_session(model, tools=tools)
            row = {
                "model": model,
                "selector": driver.model_selector(model),
                "argv": argv,
                "prompt": prompt.strip(),
                "session": launched.get("session"),
                "launch_ok": launched.get("ok"),
                "selected": launched.get("selected"),
            }
            if not launched.get("ok"):
                failures.append(
                    f"Ohmypi session for {model} did not become ready on "
                    f"{launched.get('selector')}: "
                    f"{(launched.get('pane_preview') or '')[-200:]}"
                )
                rows.append(row)
                continue
            if hasattr(driver, "send_prompt_and_wait"):
                waited = driver.send_prompt_and_wait(
                    prompt.strip(), reply_needles=reply_needles
                )
                sent = waited.get("send") or {}
                pane = str(waited.get("pane") or "")
                row["idle"] = waited.get("idle")
            else:
                sent = driver.send_keys(prompt.strip())
                row["idle"] = driver.wait_until_idle()
                pane = driver.capture_pane()
            row["send"] = sent
            if not sent.get("ok"):
                failures.append(f"tmux send-keys failed for {model}")
            row["pane_preview"] = pane[-1200:]
            send_text = f"{send_text}\n{pane}"
            selector = str(row["selector"])
            if not driver.pane_has_selector(model, pane) and selector not in pane:
                failures.append(
                    f"pane for {model} does not show selector {selector}"
                )
            exact_pong = _pane_exact_pong(pane, prompt)
            provider_404 = bool(
                provider_404_needles
                and _pane_has_any(
                    pane, provider_404_needles, prompt=prompt.strip()
                )
            )
            row["exact_pong"] = exact_pong
            row["provider_404"] = provider_404
            completed = bool(row.get("idle")) and (exact_pong or provider_404)
            row["completed"] = completed
            if not completed:
                failures.append(
                    f"TUI turn for {model} did not reach an idle exact PONG "
                    f"reply or provider 404 evidence"
                )
            rows.append(row)
            driver.close_session()
    finally:
        if hasattr(driver, "close_session"):
            driver.close_session()
    soft_fail_matches = matching_signatures(
        plan.config, text=send_text, model=list(plan.models)
    )
    warnings = [
        f"soft-fail ({row.get('name')}): {row.get('match')}" for row in soft_fail_matches
    ]
    return _with_session_history(
        plan,
        {
            "ok": not failures,
            "failures": failures,
            "warnings": warnings,
            "models": rows,
            "soft_fail_matches": soft_fail_matches,
        },
    )


def _step_tui_orchestration(plan: RunPlan, **_: Any) -> dict[str, Any]:  # noqa: PLR0915
    if not plan.tui:
        raise PlanError("orchestration kind requires --tui")
    driver = driver_for(plan.tui, plan.config)
    template = str(plan.extra.get("orchestration_prompt_template") or "")
    if not plan.orchestration_parents:
        raise PlanError("orchestration kind has no parent")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for parent in plan.orchestration_parents:
        prompt = template.replace("{parent}", parent)
        argv = driver.launch_argv(parent)
        driver.assert_no_print_flags(argv)
        rows.append(
            {
                "parent": parent,
                "children": list(plan.orchestration_children),
                "selector": driver.model_selector(parent),
                "argv": argv,
                "prompt": prompt,
            }
        )
    send_text = ""
    select = _ohmypi_select_spec(driver)
    tools = bool(select.get("tools_for_orchestration", True))
    session_started = time.time()
    try:
        first = plan.orchestration_parents[0]
        launched = driver.ensure_session(first, tools=tools)
        rows[0]["session"] = launched.get("session")
        rows[0]["launch_ok"] = launched.get("ok")
        rows[0]["selected"] = launched.get("selected")
        rows[0]["staged_agents"] = launched.get("staged_agents")
        staged = launched.get("staged_agents") or {}
        if tools and staged.get("missing"):
            failures.append(
                "Ohmypi child agent profiles missing: "
                + ", ".join(str(item) for item in staged.get("missing") or [])
            )
        if not launched.get("ok"):
            failures.append(
                f"Ohmypi session for {first} did not become ready on "
                f"{launched.get('selector')}: "
                f"{(launched.get('pane_preview') or '')[-200:]}"
            )
        else:
            orch_needles = as_str_list(select.get("orchestration_pass_needles"))
            sent = driver.send_keys(rows[0]["prompt"])
            rows[0]["send"] = sent
            if not sent.get("ok"):
                failures.append("tmux send-keys failed")
            session_dir = None
            if hasattr(driver, "spec"):
                session_dir = str((driver.spec or {}).get("session_dir") or "") or None
            wait_seconds = 420.0
            poll_seconds = 1.0
            if hasattr(driver, "_tmux_float"):
                wait_seconds = driver._tmux_float("wait_reply_seconds", 420)
                poll_seconds = driver._tmux_float("poll_interval_seconds", 1)
            deadline = time.time() + wait_seconds
            pane = driver.capture_pane() if hasattr(driver, "capture_pane") else ""
            evidence = child_spawn_evidence(
                children=list(plan.orchestration_children),
                pane=pane,
                session_dir=session_dir,
                since_mtime=session_started,
            )
            recap_present = bool(
                orch_needles
                and _pane_has_any(pane, orch_needles, prompt=rows[0]["prompt"])
            )
            # Recap is wait-complete only. Keep polling until child hub/task
            # evidence is complete even if the pane already shows recap.
            while time.time() < deadline and not evidence.get("ok"):
                time.sleep(max(poll_seconds, 0.2))
                pane = driver.capture_pane() if hasattr(driver, "capture_pane") else pane
                evidence = child_spawn_evidence(
                    children=list(plan.orchestration_children),
                    pane=pane,
                    session_dir=session_dir,
                    since_mtime=session_started,
                )
                recap_present = bool(
                    orch_needles
                    and _pane_has_any(pane, orch_needles, prompt=rows[0]["prompt"])
                )
            rows[0]["idle"] = False
            rows[0]["replied"] = bool(evidence.get("ok") or recap_present)
            rows[0]["pane_preview"] = pane[-2000:]
            send_text = pane
            selector = str(rows[0]["selector"])
            if not driver.pane_has_selector(first, pane) and selector not in pane:
                failures.append(
                    f"orchestration pane does not show selector {selector}"
                )
            rows[0]["child_evidence"] = evidence
            rows[0]["recap_present"] = recap_present
            # Recap is wait-complete only. Child hub/task evidence is the gate.
            failures.extend(list(evidence.get("failures") or []))
    finally:
        if hasattr(driver, "close_session"):
            driver.close_session()
    soft_fail_matches = _soft_fail_from_driver(
        plan,
        driver,
        list(plan.orchestration_parents) + list(plan.orchestration_children),
        extra_text=send_text,
    )
    warnings = [
        f"soft-fail ({row.get('name')}): {row.get('match')}" for row in soft_fail_matches
    ]
    return _with_session_history(
        plan,
        {
            "ok": not failures,
            "failures": failures,
            "warnings": warnings,
            "parents": rows,
            "evidence_order": _evidence_order(plan),
            "soft_fail_matches": soft_fail_matches,
        },
    )


_STEP_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "health": _step_health,
    "http_suite": _step_http_suite,
    "catalog_http": _step_catalog_http,
    "error_jsonl": _step_error_jsonl,
    "redis_scan": _step_redis,
    "docker_logs": _step_docker_logs,
    "tui_catalog": _step_tui_catalog,
    "tui_model": _step_tui_model,
    "tui_orchestration": _step_tui_orchestration,
}
