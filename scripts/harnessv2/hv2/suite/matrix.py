"""Resolve a suite selection into a stable case matrix. No TUI, no HTTP."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from hv2.artifact import git_stamp
from hv2.docker_guard import assert_container_allowed
from hv2.errors import PlanError
from hv2.instance import resolve_container_name
from hv2.load_config import as_str_list
from hv2.plan import (
    _assert_tui,
    _expand_named_models,
    _kind_spec,
    _prompt_text,
    _tui_spec,
    expand_orchestration_prompt,
)
from hv2.suite import SCHEMA_VERSION

_IMPLEMENTED = ("codex", "ohmypi", "grok", "muse")


def _sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return digest


def selection_from_args(
    *,
    tuis: Sequence[str] | None,
    kinds: Sequence[str] | None,
    models: Sequence[str] | None,
    parents: Sequence[str] | None,
    children: Sequence[str] | None,
    include_shared: bool,
) -> dict[str, Any]:
    return {
        "tuis": [str(item) for item in (tuis or [])],
        "kinds": [str(item) for item in (kinds or [])],
        "models": [str(item) for item in (models or [])],
        "parents": [str(item) for item in (parents or [])],
        "children": [str(item) for item in (children or [])] if children else None,
        "include_shared": include_shared,
    }


def _tui_names(config: Mapping[str, Any], requested: Sequence[str]) -> list[str]:
    tuis = config.get("tuis") if isinstance(config.get("tuis"), dict) else {}
    implemented = [name for name in as_str_list(tuis.get("implemented")) if name]
    if not requested:
        raise PlanError("--suite-tui is required (repeat or comma-separate)")
    names: list[str] = []
    for name in requested:
        if name == "implemented":
            names.extend(implemented)
            continue
        _assert_tui(name, config)
        spec = tuis.get(name) if isinstance(tuis.get(name), dict) else {}
        if name not in implemented or spec.get("enabled") is False or spec.get("stub") is True:
            raise PlanError(f"TUI {name!r} is not an implemented suite client")
        if name not in _IMPLEMENTED:
            raise PlanError(f"TUI {name!r} has no frozen suite contract")
        names.append(name)
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _kind_names(config: Mapping[str, Any], requested: Sequence[str]) -> list[str]:
    if not requested:
        raise PlanError("--suite-kind is required")
    out: list[str] = []
    seen: set[str] = set()
    for name in requested:
        if name not in seen:
            _kind_spec(name, config)
            if name == "platform":
                raise PlanError(
                    "platform is a shared check (--suite-shared), not a TUI case kind"
                )
            seen.add(name)
            out.append(name)
    return out


def _models_for(tui: str, kind: str, config: Mapping[str, Any], tokens: Sequence[str]) -> list[str]:
    spec = _tui_spec(tui, config)
    kind_spec = _kind_spec(kind, config)
    if tokens:
        return _expand_named_models(list(tokens), config)
    if kind == "model":
        default = spec.get("default_models") or kind_spec.get("default_models")
        if not default:
            raise PlanError(f"--suite-model is required for {tui} model cases")
        return _expand_named_models(default, config)
    if kind == "catalog":
        default = kind_spec.get("default_models") or "catalog_picker_sample"
        return _expand_named_models(default, config)
    return []


def _parents_for(
    tui: str, config: Mapping[str, Any], tokens: Sequence[str]
) -> list[str]:
    spec = _tui_spec(tui, config)
    kind_spec = _kind_spec("orchestration", config)
    if tokens:
        return _expand_named_models(list(tokens), config)
    default = (
        spec.get("default_orchestration_parent")
        or kind_spec.get("default_parent")
        or kind_spec.get("default_parent_group")
    )
    if not default:
        raise PlanError(f"--suite-parent is required for {tui} orchestration")
    return _expand_named_models(default, config)


def _children_for(
    tui: str, config: Mapping[str, Any], token: Any
) -> list[str]:
    spec = _tui_spec(tui, config)
    kind_spec = _kind_spec("orchestration", config)
    if token is None:
        if "default_orchestration_children" in spec:
            token = spec.get("default_orchestration_children")
        else:
            token = kind_spec.get("default_children_group") or "orchestration_children"
    if token in ("", None, []):
        return []
    return _expand_named_models(token, config)


def _prompt_hash(config: Mapping[str, Any], tui: str, kind: str, children: Sequence[str]) -> str:
    spec = _tui_spec(tui, config)
    if kind == "orchestration":
        name = str(spec.get("orchestration_prompt") or "orchestration")
        text = _prompt_text(config, name, {"parent": "{parent}", "home": str(Path.home())})
        text = expand_orchestration_prompt(text, parent="{parent}", children=children)
    elif kind == "model":
        name = str(spec.get("model_prompt") or "pong")
        text = _prompt_text(config, name, {"home": str(Path.home()), "repo": ""})
    else:
        text = ""
    return _sha256(text)


def _policy(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("suite") if isinstance(config.get("suite"), dict) else {}
    policy = raw.get("policy") if isinstance(raw.get("policy"), dict) else {}
    return {
        "ordering": str(policy.get("ordering") or "selection"),
        "concurrency": int(policy.get("concurrency") or 1),
        "deadline_seconds": policy.get("deadline_seconds"),
        # Unset means the driver reply wait is the case budget. A positive
        # suite.policy.case_timeout_seconds is an explicit extra halt.
        "case_timeout_seconds": policy.get("case_timeout_seconds"),
        "retry_budget": int(policy.get("retry_budget") or 0),
        "fail_fast": bool(policy.get("fail_fast") or False),
        "resume_eligible_statuses": list(
            policy.get("resume_eligible_statuses") or ["failed", "errored", "incomplete", "running"]
        ),
        "session_retention": str(policy.get("session_retention") or "retain"),
        "cleanup": str(policy.get("cleanup") or "retain_dedicated_sessions"),
    }


def _capability(tui: str, config: Mapping[str, Any]) -> dict[str, Any]:
    spec = _tui_spec(tui, config)
    select = spec.get("select_model") if isinstance(spec.get("select_model"), dict) else {}
    return {
        "tui": tui,
        "implemented": True,
        "interactive": True,
        "headless_forbidden": True,
        "pass_mode": select.get("pass_mode") or "exact_pong",
        "spawn_evidence": select.get("spawn_evidence"),
        "launch_contract": "tmux_dedicated_session",
        "readiness_contract": "ready_needles",
        "input_contract": "interactive_submit",
        "identity_contract": "session_workspace_alias",
        "terminal_evidence": select.get("spawn_evidence") or select.get("pass_mode") or "pane",
        "unsupported_reason": None,
    }


_PLATFORM_ONLY_SHARED = ("health", "http_suite", "error_jsonl", "redis_scan", "docker_logs")
_CATALOG_ONLY_SHARED = ("catalog_http", "tui_catalog")


def shared_check_ids(
    config: Mapping[str, Any], *, kinds: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Platform steps plus catalog-only steps. Never a headline case."""

    selected = set(kinds or [])
    include_platform = bool(selected - {"catalog"})
    include_catalog = "catalog" in selected
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(step_type: str) -> None:
        if step_type in seen:
            return
        seen.add(step_type)
        rows.append(
            {
                "check_id": f"shared:{step_type}",
                "type": step_type,
                "scope": "suite",
                "counts_as_case": False,
            }
        )

    if include_platform:
        for step_type in _PLATFORM_ONLY_SHARED:
            _add(step_type)
    if include_catalog:
        for step_type in _CATALOG_ONLY_SHARED:
            _add(step_type)
    return rows


def resolve_matrix(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    instance_token: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Resolve the exact case matrix. Does not inspect Docker or launch a TUI."""

    tuis = _tui_names(config, list(selection.get("tuis") or []))
    kinds = _kind_names(config, list(selection.get("kinds") or []))
    model_tokens = [str(item) for item in (selection.get("models") or [])]
    parent_tokens = [str(item) for item in (selection.get("parents") or [])]
    child_token = selection.get("children")
    container = resolve_container_name(instance_token, config)
    assert_container_allowed(container, config)
    git = git_stamp()
    meta = config.get("_meta") if isinstance(config.get("_meta"), dict) else {}
    config_hash = _sha256(
        {
            "config_path": meta.get("config_path"),
            "selection": {
                "tuis": tuis,
                "kinds": kinds,
                "models": model_tokens,
                "parents": parent_tokens,
                "children": child_token,
            },
        }
    )
    identity = {
        "source_commit": git.get("commit"),
        "source_branch": git.get("branch"),
        "source_dirty": git.get("dirty"),
        "config_path": meta.get("config_path"),
        "config_hash": config_hash,
        "container": container,
        "instance_token": instance_token or str(config.get("default_instance")),
        "runtime": "unresolved" if dry_run else "pending_inspect",
        "target_base_url": None,
        "forbidden_targets": ["aawm-litellm", "litellm-dev", 4000, 4001],
    }
    run_id = _stable_id(
        SCHEMA_VERSION,
        config_hash,
        str(git.get("commit") or ""),
        ",".join(tuis),
        ",".join(kinds),
    )
    cases: list[dict[str, Any]] = []
    for tui in tuis:
        for kind in kinds:
            if kind == "orchestration":
                parents = _parents_for(tui, config, parent_tokens)
                children = _children_for(tui, config, child_token)
                prompt_hash = _prompt_hash(config, tui, kind, children)
                for parent in parents:
                    case_id = _stable_id(run_id, tui, kind, parent)
                    cases.append(
                        {
                            "case_id": case_id,
                            "run_id": run_id,
                            "tui": tui,
                            "kind": kind,
                            "scenario": "orchestration",
                            "model": None,
                            "parent": parent,
                            "children": list(children),
                            "alias": parent,
                            "attempt_id": f"{case_id}:1",
                            "attempt": 1,
                            "prompt_hash": prompt_hash,
                            "counts_as_case": True,
                            "status": "planned",
                        }
                    )
            elif kind == "model":
                models = _models_for(tui, kind, config, model_tokens)
                prompt_hash = _prompt_hash(config, tui, kind, [])
                for model in models:
                    case_id = _stable_id(run_id, tui, kind, model)
                    cases.append(
                        {
                            "case_id": case_id,
                            "run_id": run_id,
                            "tui": tui,
                            "kind": kind,
                            "scenario": "model",
                            "model": model,
                            "parent": None,
                            "children": [],
                            "alias": model,
                            "attempt_id": f"{case_id}:1",
                            "attempt": 1,
                            "prompt_hash": prompt_hash,
                            "counts_as_case": True,
                            "status": "planned",
                        }
                    )
            elif kind == "catalog":
                continue
    shared = (
        shared_check_ids(config, kinds=kinds)
        if selection.get("include_shared", True)
        else []
    )
    return {
        "schema": SCHEMA_VERSION,
        "run_id": run_id,
        "dry_run": dry_run,
        "identity": identity,
        "policy": _policy(config),
        "capabilities": [_capability(tui, config) for tui in tuis],
        "selection": {
            "tuis": tuis,
            "kinds": kinds,
            "models": model_tokens,
            "parents": parent_tokens,
            "children": child_token,
        },
        "cases": cases,
        "shared_checks": shared,
        "planned_cases": len(cases),
        "planned_shared_checks": len(shared),
        "child_assertion_slots": sum(len(case["children"]) for case in cases),
        "stable": True,
    }
