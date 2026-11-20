"""Render a static side-by-side AAWM alias-routing comparison.

The renderer consumes compiled routing snapshots and keeps the output
self-contained so it can be opened offline from an artifact or from disk.  It
renders each alias's directly configured entries in order and links references
to their target definitions instead of expanding every alias path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from html import escape

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
    AliasReference,
    RoutingAlias,
    RoutingCandidate,
    RoutingSnapshot,
    ScheduleWindow,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
    _ANTHROPIC_CREDENTIAL_ROUTE_FAMILIES,
    _order_snapshot_entries_by_priority,
)


_WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True)
class _ReportRow:
    """One directly configured routing entry at its own priority."""

    kind: str
    priority: int
    own_restrictions: tuple[str, ...]
    inherited_restrictions: tuple[str, ...] = ()
    target_alias: str | None = None
    target_exists: bool = False
    provider: str | None = None
    model: str | None = None
    route_family: str | None = None
    anthropic_route_family: str | None = None
    reasoning_effort: str | None = None
    schedule: ScheduleWindow | None = None

    @property
    def label(self) -> str:
        if self.target_alias is not None:
            return self.target_alias
        return f"{self.provider} / {self.model}"

    @property
    def restrictions(self) -> tuple[str, ...]:
        return (*self.inherited_restrictions, *self.own_restrictions)


@dataclass(frozen=True)
class _ReportAlias:
    """A distinct alias and its directly configured rows in each environment."""

    anchor: str
    alias_name: str
    alpha: RoutingAlias | None
    production: RoutingAlias | None
    alpha_rows: tuple[_ReportRow, ...]
    production_rows: tuple[_ReportRow, ...]


def _configured_rows(
    snapshot: RoutingSnapshot,
    alias_name: str,
    *,
    inherited_restrictions: tuple[str, ...] = (),
) -> tuple[_ReportRow, ...]:
    """Return the alias's configured list without expanding references."""

    alias = snapshot.aliases.get(alias_name)
    if alias is None or alias.dispatch is not None:
        return ()
    rows: list[_ReportRow] = []
    for entry in _order_snapshot_entries_by_priority(alias.candidates):
        own_restrictions = _entry_restrictions(entry)
        if isinstance(entry, RoutingCandidate):
            rows.append(
                _ReportRow(
                    kind="model",
                    priority=entry.priority,
                    own_restrictions=own_restrictions,
                    inherited_restrictions=inherited_restrictions,
                    provider=entry.provider,
                    model=entry.model,
                    route_family=entry.route_family,
                    anthropic_route_family=entry.anthropic_route_family,
                    reasoning_effort=entry.reasoning_effort,
                    schedule=entry.schedule,
                )
            )
            continue
        rows.append(
            _ReportRow(
                kind="alias reference",
                priority=entry.priority,
                own_restrictions=own_restrictions,
                inherited_restrictions=inherited_restrictions,
                target_alias=entry.alias_name,
                target_exists=entry.alias_name in snapshot.aliases,
                schedule=entry.schedule,
            )
        )
    return tuple(rows)


def _direct_condition_groups(snapshot: RoutingSnapshot, alias_name: str) -> tuple[tuple[str, ...], ...]:
    """Return distinct gate sets attached directly to a target alias's rows."""

    alias = snapshot.aliases.get(alias_name)
    if alias is None or alias.dispatch is not None:
        return ()
    return tuple(dict.fromkeys(_entry_restrictions(entry) for entry in alias.candidates))


def _combined_reference_conditions(snapshot: RoutingSnapshot, row: _ReportRow) -> tuple[str, ...]:
    """Combine inherited, reference, and direct target gates for one linked row."""

    conditions = [*row.inherited_restrictions, *row.own_restrictions]
    if row.target_alias is None or row.target_alias not in snapshot.aliases:
        return tuple(conditions)
    for group in _direct_condition_groups(snapshot, row.target_alias):
        if not group:
            conditions.append(f"target has an ungated row: {row.target_alias}")
        else:
            conditions.append(f"target row applies: {'; '.join(group)}")
    return tuple(conditions)


def _entry_restrictions(entry: RoutingCandidate | AliasReference) -> tuple[str, ...]:
    restrictions: list[str] = []
    if entry.tui_attached is not None:
        restrictions.append(f"client attached: {entry.tui_attached}")
    if entry.tui_excluded is not None:
        restrictions.append(f"client excluded: {entry.tui_excluded}")
    if entry.schedule is not None:
        restrictions.append(f"schedule gate: {_schedule_label(entry.schedule)}")
    return tuple(restrictions)


def _format_offset(offset: timedelta) -> str:
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    minutes = abs(total_minutes)
    return f"{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def _schedule_timezone(schedule: ScheduleWindow) -> str:
    if schedule.timezone is not None:
        return schedule.timezone
    if schedule.utc_offset is not None:
        return f"UTC{_format_offset(schedule.utc_offset)}"
    return "UTC"


def _schedule_label(schedule: ScheduleWindow) -> str:
    """Describe a compiled window using the runtime's inclusive/exclusive rules."""

    if schedule.kind == "absolute":
        start = schedule.start.isoformat() if schedule.start is not None else "missing"
        end = schedule.end.isoformat() if schedule.end is not None else "missing"
        return f"absolute {start} through {end}, inclusive endpoints"

    timezone_label = _schedule_timezone(schedule)
    date_parts: list[str] = []
    if schedule.start_date is not None and schedule.end_date is not None:
        date_parts = [
            f"dates {schedule.start_date.isoformat()} through {schedule.end_date.isoformat()}, inclusive",
        ]

    if schedule.start_time is None or schedule.end_time is None:
        weekday_label = "all days" if schedule.weekdays is None else _weekday_label(schedule.weekdays)
        return "; ".join([f"daily all day ({timezone_label})", weekday_label, *date_parts])

    start = schedule.start_time.isoformat()
    end = schedule.end_time.isoformat()
    start_seconds = _clock_seconds(schedule.start_time)
    end_seconds = _clock_seconds(schedule.end_time)
    if start_seconds == end_seconds:
        semantics = "inactive (equal start and end)"
    elif start_seconds > end_seconds:
        semantics = "start inclusive, end exclusive; wraps across midnight"
    else:
        semantics = "start inclusive, end exclusive"
    weekday_label = "all days" if schedule.weekdays is None else _weekday_label(schedule.weekdays)
    return "; ".join([f"daily {start} to {end} ({timezone_label})", semantics, weekday_label, *date_parts])


def _clock_seconds(value: time) -> int:
    return value.hour * 3600 + value.minute * 60 + value.second


def _weekday_label(weekdays: tuple[int, ...]) -> str:
    return ", ".join(_WEEKDAY_NAMES[weekday] for weekday in weekdays)


def _anchor_for(alias_name: str) -> str:
    """Encode every byte so visually similar names cannot share an anchor."""

    return f"alias-{alias_name.encode('utf-8').hex()}"


def _alias_link(snapshot: RoutingSnapshot, alias_name: str) -> str:
    if alias_name not in snapshot.aliases:
        return escape(alias_name)
    return f'<a href="#{escape(_anchor_for(alias_name))}">{escape(alias_name)}</a>'


def _restriction_list(conditions: tuple[str, ...]) -> str:
    if not conditions:
        return "No inherited, schedule, client, or linked-target gate"
    items = "".join(f"<li>{escape(condition)}</li>" for condition in conditions)
    return f"<ul>{items}</ul>"


def _dispatch_definition(snapshot: RoutingSnapshot, alias: RoutingAlias | None) -> str:
    if alias is None or alias.dispatch is None:
        return "No dispatch projection"
    dispatch = alias.dispatch
    rows = "".join(
        f'<tr><th scope="row">{escape(rule.tui_family)}</th><td>{_alias_link(snapshot, rule.target_alias)}</td></tr>'
        for rule in dispatch.by_tui
    )
    default = "Not configured"
    if dispatch.default is not None:
        default = _alias_link(snapshot, dispatch.default)
    blocked = ", ".join(dispatch.blocked_tui_families) if dispatch.blocked_tui_families else "None"
    if rows:
        rules = (
            "<table><caption>Client-family rules</caption><thead><tr>"
            '<th scope="col">Client family</th><th scope="col">Target</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>"
        )
    else:
        rules = "<p>No explicit client-family rules.</p>"
    return (
        f"{rules}<dl><div><dt>Default</dt><dd>{default}</dd></div>"
        f"<div><dt>Blocked client families</dt><dd>{escape(blocked)}</dd></div></dl>"
    )


def _route_projection(row: _ReportRow) -> str:
    if row.kind != "model":
        return "Linked alias definition"
    codex_target = (
        "unavailable" if row.route_family in _ANTHROPIC_CREDENTIAL_ROUTE_FAMILIES else row.route_family or "unavailable"
    )
    anthropic_target = row.anthropic_route_family or "unavailable"
    return f"Codex: {codex_target}; Anthropic: {anthropic_target}"


def _row_item(snapshot: RoutingSnapshot, row: _ReportRow, index: int) -> str:
    order = "priority zero (last resort)" if row.priority == 0 else f"priority {row.priority}"
    target = _alias_link(snapshot, row.target_alias) if row.target_alias is not None else escape(row.label)
    kind = "missing alias reference" if row.kind == "alias reference" and not row.target_exists else row.kind
    conditions = _combined_reference_conditions(snapshot, row) if row.kind == "alias reference" else row.restrictions
    details = []
    if row.reasoning_effort is not None:
        details.append(f"<div><dt>Reasoning effort</dt><dd>{escape(row.reasoning_effort)}</dd></div>")
    if row.schedule is not None:
        details.append(f"<div><dt>Schedule</dt><dd>{escape(_schedule_label(row.schedule))}</dd></div>")
    return (
        f"<li><article><h4>{index}. {target}</h4>"
        f"<dl><div><dt>Kind</dt><dd>{escape(kind)}</dd></div>"
        f"<div><dt>Order</dt><dd>{escape(order)}</dd></div>"
        f"<div><dt>Ingress projection</dt><dd>{escape(_route_projection(row))}</dd></div>"
        f"<div><dt>Restrictions</dt><dd>{_restriction_list(conditions)}</dd></div>{''.join(details)}</dl>"
        "</article></li>"
    )


def _row_list(snapshot: RoutingSnapshot, rows: tuple[_ReportRow, ...], environment: str) -> str:
    if not rows:
        return f"<p>No configured routing entries for {escape(environment)}.</p>"
    items = "".join(_row_item(snapshot, row, index) for index, row in enumerate(rows, 1))
    return f'<ol class="entries">{items}</ol>'


def _environment_rows(
    alpha: RoutingSnapshot,
    production: RoutingSnapshot,
    *,
    alpha_readiness_hash: str | None,
    production_readiness_hash: str | None,
) -> str:
    snapshots = {
        "alpha": (alpha, alpha_readiness_hash),
        "production": (production, production_readiness_hash),
    }
    rows = ""
    for environment, (snapshot, readiness_hash) in snapshots.items():
        if readiness_hash is None:
            readiness = "unavailable"
        elif readiness_hash == snapshot.config_hash:
            readiness = "matched"
        else:
            readiness = "mismatched"
        rows += (
            f'<tr><th scope="row">{escape(environment.capitalize())}</th>'
            f"<td><code>{escape(snapshot.config_hash)}</code></td>"
            f"<td><code>{escape(snapshot.config_version)}</code></td>"
            f"<td>{escape(readiness)}</td><td>{len(snapshot.aliases)}</td></tr>"
        )
    return rows


def _alias_comparison(
    report_alias: _ReportAlias,
    snapshots: dict[str, RoutingSnapshot],
) -> str:
    alpha = snapshots["alpha"]
    production = snapshots["production"]
    alpha_definition = "Not configured" if report_alias.alpha is None else escape(report_alias.alpha.name)
    production_definition = (
        "Not configured" if report_alias.production is None else escape(report_alias.production.name)
    )
    alpha_strategy = report_alias.alpha.distribution_strategy if report_alias.alpha is not None else None
    production_strategy = report_alias.production.distribution_strategy if report_alias.production is not None else None
    strategy_display = escape(f"{alpha_strategy or 'configured order'} / {production_strategy or 'configured order'}")
    alpha_dispatch = _dispatch_definition(alpha, report_alias.alpha)
    production_dispatch = _dispatch_definition(production, report_alias.production)
    alpha_rows = _row_list(alpha, report_alias.alpha_rows, "alpha")
    production_rows = _row_list(production, report_alias.production_rows, "production")
    return (
        f"<table><caption>{escape(report_alias.alias_name)} comparison</caption><thead><tr>"
        '<th scope="col">Aspect</th><th scope="col">Alpha</th>'
        '<th scope="col">Production</th></tr></thead><tbody>'
        f'<tr><th scope="row">Definition</th><td>{alpha_definition}</td><td>{production_definition}</td></tr>'
        f'<tr><th scope="row">Distribution</th><td colspan="2">{strategy_display}</td></tr>'
        f'<tr><th scope="row">Dispatch</th><td>{alpha_dispatch}</td><td>{production_dispatch}</td></tr>'
        '<tr><th scope="row">Configured entries</th>'
        f"<td>{alpha_rows}</td>"
        f"<td>{production_rows}</td></tr>"
        "</tbody></table>"
    )


def render_report(
    alpha: RoutingSnapshot,
    production: RoutingSnapshot,
    *,
    alpha_readiness_hash: str | None,
    production_readiness_hash: str | None,
) -> str:
    """Render an accessible offline HTML comparison for two routing snapshots."""

    snapshots = {"alpha": alpha, "production": production}
    report_aliases: list[_ReportAlias] = []
    for alias_name in sorted(dict.fromkeys((*alpha.aliases, *production.aliases))):
        report_aliases.append(
            _ReportAlias(
                anchor=_anchor_for(alias_name),
                alias_name=alias_name,
                alpha=alpha.aliases.get(alias_name),
                production=production.aliases.get(alias_name),
                alpha_rows=_configured_rows(alpha, alias_name) if alias_name in alpha.aliases else (),
                production_rows=(_configured_rows(production, alias_name) if alias_name in production.aliases else ()),
            )
        )

    navigation_items = "".join(
        f'<li><a href="#{escape(report_alias.anchor)}">{escape(report_alias.alias_name)}</a></li>'
        for report_alias in report_aliases
    )
    articles = "".join(
        f'<article id="{escape(report_alias.anchor)}" aria-labelledby="heading-{escape(report_alias.anchor)}">'
        f'<h3 id="heading-{escape(report_alias.anchor)}">{escape(report_alias.alias_name)}</h3>'
        f"{_alias_comparison(report_alias, snapshots)}</article>"
        for report_alias in report_aliases
    )
    environment_rows = _environment_rows(
        alpha,
        production,
        alpha_readiness_hash=alpha_readiness_hash,
        production_readiness_hash=production_readiness_hash,
    )
    generated = datetime.now(timezone.utc).isoformat()
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>AAWM alias routing comparison</title><style>"
        ":root { color-scheme: light dark; font-family: system-ui, sans-serif; }"
        "body { max-width: 140rem; margin: auto; padding: 1rem; }"
        "table { border-collapse: collapse; width: 100%; }"
        "caption, th, td { border: 1px solid currentColor; padding: .5rem; text-align: left; }"
        "dl, ul, ol { margin: .25rem 0; padding-left: 1.25rem; }"
        "dt { font-weight: 700; } code { overflow-wrap: anywhere; }"
        "@media (max-width: 60rem) { table, thead, tbody, tr, td, th { display: block; } }"
        "</style></head><body><main><h1>AAWM alias routing comparison</h1>"
        f"<p>Generated {escape(generated)}.</p>"
        '<section aria-labelledby="environment-summary"><h2 id="environment-summary">'
        "Environment summary</h2><table><caption>Snapshot identity and readiness</caption>"
        '<thead><tr><th scope="col">Environment</th><th scope="col">Compiled hash</th>'
        '<th scope="col">Version</th><th scope="col">Readiness</th>'
        '<th scope="col">Alias definitions</th></tr></thead>'
        f"<tbody>{environment_rows}</tbody></table></section>"
        '<nav aria-labelledby="alias-navigation"><h2 id="alias-navigation">Aliases</h2>'
        f"<ol>{navigation_items}</ol></nav>"
        '<section aria-labelledby="alias-comparisons"><h2 id="alias-comparisons">Alias comparison</h2>'
        f"{articles}</section></main></body></html>"
    )
