"""CLI + config → RunPlan. Fail closed before any docker/HTTP work."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from hv2.cli import split_csv
from hv2.docker_guard import assert_container_allowed
from hv2.errors import PlanError
from hv2.instance import ResolvedInstance, resolve_container_name
from hv2.load_config import as_str_list, expand_string


@dataclass(frozen=True)
class RunPlan:
    kind: str
    tui: str | None
    instance_token: str
    container: str
    models: tuple[str, ...]
    orchestration_parents: tuple[str, ...]
    orchestration_children: tuple[str, ...]
    requires_tui: bool
    dry_run: bool
    write_artifact: Path | None
    steps: tuple[dict[str, Any], ...]
    config: dict[str, Any]
    resolved: ResolvedInstance | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "tui": self.tui,
            "instance": self.instance_token,
            "container": self.container,
            "models": list(self.models),
            "orchestration_parents": list(self.orchestration_parents),
            "orchestration_children": list(self.orchestration_children),
            "requires_tui": self.requires_tui,
            "dry_run": self.dry_run,
            "write_artifact": str(self.write_artifact) if self.write_artifact else None,
            "steps": [dict(step) for step in self.steps],
        }
        if "tools_for_orchestration" in self.extra:
            payload["tools_for_orchestration"] = bool(
                self.extra.get("tools_for_orchestration")
            )
        prompt = self.extra.get("orchestration_prompt_template")
        if prompt:
            payload["orchestration_prompt"] = str(prompt)
        if self.resolved is not None:
            payload["base_url"] = self.resolved.base_url
            payload["host_port"] = self.resolved.host_port
        return payload


def _kinds(config: Mapping[str, Any]) -> dict[str, Any]:
    kinds = config.get("kinds")
    if not isinstance(kinds, dict) or not kinds:
        raise PlanError("kinds.yaml did not load a kinds mapping")
    return kinds


def _tuis(config: Mapping[str, Any]) -> dict[str, Any]:
    tuis = config.get("tuis")
    if not isinstance(tuis, dict):
        raise PlanError("tuis.yaml did not load a tuis mapping")
    return tuis


def _models_block(config: Mapping[str, Any]) -> dict[str, Any]:
    models = config.get("models")
    if not isinstance(models, dict):
        raise PlanError("models.yaml did not load a models mapping")
    return models


def skip_prefixes(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        prefix
        for prefix in as_str_list(_models_block(config).get("skip_prefixes"))
        if prefix
    )


def _has_skip_prefix(model_id: str, prefixes: Sequence[str]) -> bool:
    return any(model_id.startswith(prefix) for prefix in prefixes)


def _drop_skipped_ids(ids: Sequence[str], prefixes: Sequence[str]) -> list[str]:
    return [item for item in ids if not _has_skip_prefix(item, prefixes)]


def _raw_compiled_aliases(config: Mapping[str, Any]) -> list[str]:
    return as_str_list(_models_block(config).get("compiled_aliases"))


def compiled_aliases(config: Mapping[str, Any]) -> list[str]:
    return _drop_skipped_ids(_raw_compiled_aliases(config), skip_prefixes(config))


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def expand_group(name: str, config: Mapping[str, Any]) -> list[str]:
    return _expand_group(name, config, skip_prefixes(config))


def _expand_group(
    name: str, config: Mapping[str, Any], prefixes: Sequence[str]
) -> list[str]:
    block = _models_block(config)
    compiled = _drop_skipped_ids(_raw_compiled_aliases(config), prefixes)
    if name in compiled:
        return [name]
    groups = block.get("groups") if isinstance(block.get("groups"), dict) else {}
    if name in groups:
        value = groups[name]
        if isinstance(value, str):
            return _expand_group(value, config, prefixes)
        out: list[str] = []
        for item in as_str_list(value):
            if item == "compiled_aliases" or item in groups or item in block:
                out.extend(_expand_group(item, config, prefixes))
            else:
                out.append(item)
        return _unique(_drop_skipped_ids(out, prefixes))
    if name == "compiled_aliases":
        return compiled
    sample = block.get(name)
    if isinstance(sample, list):
        return _drop_skipped_ids(as_str_list(sample), prefixes)
    # Unknown token is treated as a concrete model id (operator overlay).
    return _drop_skipped_ids([name], prefixes)


def _skipped_model_error(token: str) -> PlanError:
    return PlanError(
        f"explicit --model {token!r} is not allowed by models.skip_prefixes"
    )


def _expand_model_args(tokens: Sequence[str], config: Mapping[str, Any]) -> list[str]:
    # Group expansion silently drops skip-prefix ids. An explicit --model token
    # that starts with a skip prefix, or that expands only to skipped ids,
    # fails closed.
    prefixes = skip_prefixes(config)
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if _has_skip_prefix(token, prefixes):
            raise _skipped_model_error(token)
        expanded = _expand_group(token, config, prefixes)
        if not expanded:
            raise _skipped_model_error(token)
        for item in expanded:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _assert_tui(name: str, config: Mapping[str, Any]) -> None:
    tuis = _tuis(config)
    out_of_scope = set(as_str_list(tuis.get("out_of_scope")))
    if name in out_of_scope or name == "claude":
        raise PlanError(
            f"TUI {name!r} is out of scope for harness v2. Claude stays in local-ci."
        )
    implemented = set(as_str_list(tuis.get("implemented")))
    stubs = set(as_str_list(tuis.get("stubs")))
    known = (implemented | stubs | set(tuis.keys())) - {
        "out_of_scope",
        "implemented",
        "stubs",
    }
    if name not in known:
        raise PlanError(f"unknown TUI {name!r}; known: {sorted(known)}")
    spec = tuis.get(name) if isinstance(tuis.get(name), dict) else {}
    if name in stubs or spec.get("stub") is True or spec.get("enabled") is False:
        if name not in implemented:
            raise PlanError(f"TUI driver {name!r} is not implemented")
    forbid = as_str_list(spec.get("forbid_flags"))
    argv_keys = (
        "argv_interactive",
        "argv_launch_model",
        "argv_no_tools",
        "catalog_json_argv",
        "catalog_find_argv",
    )
    forbid_tokens = as_str_list(spec.get("forbid_tokens"))
    for key in argv_keys:
        argv = as_str_list(spec.get(key))
        for flag in (*forbid, *forbid_tokens):
            if flag and flag in argv:
                raise PlanError(
                    f"TUI {name!r} argv contains forbidden flag {flag!r} ({key})"
                )


def _tui_spec(tui: str | None, config: Mapping[str, Any]) -> dict[str, Any]:
    if not tui:
        return {}
    spec = _tuis(config).get(tui)
    return spec if isinstance(spec, dict) else {}


def _expand_named_models(raw: Any, config: Mapping[str, Any]) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return _expand_model_args([str(item) for item in raw], config)
    if raw:
        return expand_group(str(raw), config)
    return []


def _kind_spec(kind: str, config: Mapping[str, Any]) -> dict[str, Any]:
    kinds = _kinds(config)
    if kind not in kinds:
        raise PlanError(
            f"unknown --test {kind!r}; known: {sorted(str(k) for k in kinds)}"
        )
    spec = kinds[kind]
    if not isinstance(spec, dict):
        raise PlanError(f"kind {kind!r} must be a mapping")
    return spec


def render_orchestration_children_block(children: Sequence[str]) -> str:
    """Render the Ohmypi `agent=` spawn list from the planned children.

    The first child carries the PONG-then-date contract; later children
    reuse that task. An empty children list yields an empty block.
    """

    if not children:
        return ""
    first = str(children[0])
    lines = [
        f"1. agent={first}   FIRST: reply with exactly the word PONG. Then run the",
        "   single shell command `date` (or `date -u` if date is unavailable).",
        "   Return only the command stdout. Do not guess the time.",
    ]
    for index, name in enumerate(children[1:], start=2):
        lines.append(f"{index}. agent={name} same task.")
    return "\n".join(lines)


def expand_orchestration_prompt(
    template: str,
    *,
    parent: str,
    children: Sequence[str],
) -> str:
    """Substitute parent/children placeholders in an orchestration template."""

    return expand_string(
        template,
        {
            "parent": parent,
            "child_count": str(len(children)),
            "children_block": render_orchestration_children_block(children),
        },
    )


def _prompt_text(config: Mapping[str, Any], name: str, context: Mapping[str, str]) -> str:
    prompts = config.get("prompts") if isinstance(config.get("prompts"), dict) else {}
    entry = prompts.get(name)
    if isinstance(entry, dict) and entry.get("file"):
        path = Path(expand_string(str(entry["file"]), context))
        if not path.is_absolute():
            harness_dir = Path(__file__).resolve().parents[1]
            path = harness_dir / "config" / path
        if not path.is_file():
            raise PlanError(f"prompt file missing: {path}")
        return expand_string(path.read_text(encoding="utf-8"), context)
    if isinstance(entry, str):
        return expand_string(entry, context)
    raise PlanError(f"prompt {name!r} is not defined")


def build_plan(  # noqa: PLR0915
    *,
    config: Mapping[str, Any],
    kind: str,
    instance_token: str | None,
    tui: str | None,
    models: Sequence[str] | None,
    orchestration_parent: str | None,
    orchestration_children: str | None,
    dry_run: bool,
    write_artifact: Path | None,
) -> RunPlan:
    spec = _kind_spec(kind, config)
    requires_tui = bool(spec.get("requires_tui"))
    tui_optional = bool(spec.get("tui_optional"))
    if requires_tui and not tui:
        raise PlanError(f"--tui is required for --test {kind}")
    if tui and not requires_tui and not tui_optional:
        raise PlanError(f"--tui is forbidden for --test {kind}")
    if tui:
        _assert_tui(tui, config)

    container = resolve_container_name(instance_token, config)
    assert_container_allowed(container, config)

    model_tokens = split_csv(list(models or []))
    selected_models: list[str] = []
    parents: list[str] = []
    children: list[str] = []
    extra: dict[str, Any] = {}
    tui_spec = _tui_spec(tui, config)

    if kind == "model":
        require_explicit_model = bool(spec.get("require_explicit_model"))
        if require_explicit_model and not model_tokens:
            raise PlanError("--model is required")
        if model_tokens:
            selected_models = _expand_model_args(model_tokens, config)
        else:
            default_group = (
                tui_spec.get("default_models")
                or spec.get("default_models")
                or _models_block(config).get("default_model_group")
            )
            if not default_group:
                raise PlanError("--model is required when no default group is set")
            selected_models = _expand_named_models(default_group, config)
        prompt_name = str(tui_spec.get("model_prompt") or "pong")
        extra["pong_prompt"] = _prompt_text(
            config, prompt_name, {"home": str(Path.home()), "repo": ""}
        )
    elif kind == "catalog":
        default_group = spec.get("default_models") or "catalog_picker_sample"
        if model_tokens:
            selected_models = _expand_model_args(model_tokens, config)
        else:
            selected_models = expand_group(str(default_group), config)
    elif kind == "orchestration":
        default_parent = spec.get("default_parent")
        default_parent_group = spec.get("default_parent_group")
        default_orchestration_parent = _models_block(config).get(
            "default_orchestration_parent"
        )
        parent_token = (
            orchestration_parent
            or tui_spec.get("default_orchestration_parent")
            or default_parent
            or default_parent_group
            or default_orchestration_parent
        )
        if not parent_token:
            raise PlanError("--orchestration-parent is required")
        parents = _expand_named_models(parent_token, config)
        child_token = (
            orchestration_children
            or tui_spec.get("default_orchestration_children")
            or spec.get("default_children_group")
            or "orchestration_children"
        )
        children = _expand_named_models(child_token, config)
        prompt_name = str(tui_spec.get("orchestration_prompt") or "orchestration")
        raw_template = _prompt_text(
            config, prompt_name, {"parent": "{parent}", "home": str(Path.home())}
        )
        extra["orchestration_prompt_template"] = expand_orchestration_prompt(
            raw_template,
            parent="{parent}",
            children=children,
        )
        extra["tools_for_orchestration"] = bool(
            (tui_spec.get("select_model") or {}).get("tools_for_orchestration", True)
            if isinstance(tui_spec.get("select_model"), dict)
            else True
        )
    elif kind == "platform":
        selected_models = []
    else:
        if model_tokens:
            selected_models = _expand_model_args(model_tokens, config)

    steps = spec.get("steps") or []
    if not isinstance(steps, list):
        raise PlanError(f"kind {kind!r} steps must be a list")
    normalized_steps = [dict(step) for step in steps if isinstance(step, dict)]

    return RunPlan(
        kind=kind,
        tui=tui,
        instance_token=instance_token or str(config.get("default_instance")),
        container=container,
        models=tuple(selected_models),
        orchestration_parents=tuple(parents),
        orchestration_children=tuple(children),
        requires_tui=requires_tui,
        dry_run=dry_run,
        write_artifact=write_artifact,
        steps=tuple(normalized_steps),
        config=dict(config),
        extra=extra,
    )
