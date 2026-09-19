#!/usr/bin/env python3
"""Convert LiteLLM completed*.md ledger items into archived aawmdt todo records.

Canonical copy: scripts/convert_completed_ledgers_to_aawmdt_todo.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / ".analysis"
SCRATCH = Path(
    __import__("os").environ.get(
        "AAWMDT_CONVERSION_SCRATCH",
        str(ANALYSIS / "aawmdt-conversion"),
    )
)
AAWMDT = "aawmdt"
ACTOR = "completed-ledger-conversion"

HYPHEN_PREFIX_MAP = {
    "ALI-QUOTA": "ALIQUOTA",
    "ALI-PRICING": "ALIPRICING",
    "ALI-PROD": "ALIPROD",
    "ALI-NC": "ALINC",
    "MS-QUOTA": "MSQUOTA",
    "MS-PROD": "MSPROD",
    "P-INV": "PINV",
}

MODEL_PREFIXES = {"GPT", "GLM", "KIMI", "CLAUDE", "GROK"}
SKIP_HEADING_RE = re.compile(
    r"^(?:"
    r"Completed(?:\s+Work)?(?:\s*[-:]\s*.*)?|"
    r"Completed TODOs|"
    r"Ledger Split Direction|"
    r"Source closeout.*|"
    r"Latest published source.*|"
    r"Goal|Result|Evidence|Verification|Breakdown|"
    r"Changed tracked paths|Exact verification commands and results|"
    r"Runtime evidence|Artifact and transcripts|Reopened disposition|"
    r"Published commits|Residual boundary|Runtime boundary|"
    r"Outcome|Item reconciliation|Live acceptance evidence|"
    r"Residual release boundaries|Post-completion correction.*|"
    r"Why a new table instead of an existing one|"
    r"Scope controls|Selected work(?: \(source\))?|Delivery|Reviews|"
    r"Worklog|Acceptance evidence|Boundaries|Current state|"
    r"Follow-ups?|Hazards|Notes|Residual(?: scope)?|"
    r"Publication(?: and deployment)?|Changed paths|"
    r"Implementation(?: summary| notes)?|Sibling handoff|"
    r"Archived source files.*|Disposition|Status|"
    r"\d{4}-\d{2}-\d{2}(?:\s+Late(?: Completed Items)?)?|"
    r"\d{4}-\d{2}-\d{2} Late|"
    r"\d{4}-\d{2}-\d{2} Completed Work"
    r")$",
    re.I,
)
DATE_HEADING_RE = re.compile(r"^#{1,3}\s+(\d{4}-\d{2}-\d{2})\s*$")
FILE_TITLE_RE = re.compile(
    r"^#\s+(Completed|Ledger).*$",
    re.I,
)
HEADING_RE = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
FULL_ID_RE = re.compile(
    r"\b([A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]+)*)-(\d+)\b"
)
THROUGH_RE = re.compile(
    r"\b([A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]+)*)-(\d+)\s+through\s+"
    r"(?:([A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]+)*)-)?(\d+)\b",
    re.I,
)
BULLET_ITEM_RE = re.compile(
    r"^(\s*)-\s+([A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]+)*)-(\d+)\b(.*)$"
)
ANON_BULLET_RE = re.compile(r"^- \[[ xX]\] ")
TZ_OFFSETS = {
    "UTC": timezone.utc,
    "Z": timezone.utc,
    "EDT": timezone(timedelta(hours=-4)),
    "EST": timezone(timedelta(hours=-5)),
    "CDT": timezone(timedelta(hours=-5)),
    "CST": timezone(timedelta(hours=-6)),
    "PDT": timezone(timedelta(hours=-7)),
    "PST": timezone(timedelta(hours=-8)),
}


def ledger_files() -> list[Path]:
    files = list(ANALYSIS.glob("completed*.md")) + list((ANALYSIS / "completed").glob("completed*.md"))
    return sorted({p.resolve() for p in files if p.is_file()})


def queue_files() -> list[Path]:
    files = [ANALYSIS / "todo.md", *ANALYSIS.glob("*.todo.md")]
    return sorted({p.resolve() for p in files if p.is_file() and p.name != "todo.deferred.md"})


def is_date_sequence(sequence: int) -> bool:
    text = str(sequence)
    return len(text) == 8 and text.startswith("20")


def cli_prefix(raw: str, sequence: int | None = None) -> str:
    raw = raw.upper()
    mapped = HYPHEN_PREFIX_MAP.get(raw, raw.replace("-", ""))
    if sequence is not None and is_date_sequence(sequence):
        return f"{mapped}DATE"
    return mapped


def is_model_id(prefix: str, sequence: str, rest: str) -> bool:
    if prefix in MODEL_PREFIXES and rest.startswith("."):
        return True
    if prefix in MODEL_PREFIXES and len(sequence) <= 2:
        return True
    if prefix == "UTF" and sequence == "8":
        return True
    return False


def heading_id_region(title: str) -> str:
    """Keep only the leading ID list; ignore TAP/model mentions in the title."""
    stripped = re.sub(r"^\d{4}-\d{2}-\d{2}\s+", "", title.strip())
    split = re.split(r"\s+[-—]\s+", stripped, maxsplit=1)
    return split[0]


def expand_ids(text: str) -> list[tuple[str, int, str]]:
    """Return (raw_prefix, sequence, source_id) preserving heading order."""
    found: list[tuple[str, int, str]] = []
    seen: set[str] = set()

    def add(prefix: str, seq: int) -> None:
        source_id = f"{prefix}-{seq}"
        key = f"{cli_prefix(prefix, seq)}-{seq}"
        if key in seen:
            return
        if is_model_id(prefix, str(seq), ""):
            return
        seen.add(key)
        found.append((prefix.upper(), seq, source_id))

    working = text
    for match in THROUGH_RE.finditer(text):
        start_prefix = match.group(1).upper()
        start = int(match.group(2))
        end_prefix = (match.group(3) or start_prefix).upper()
        end = int(match.group(4))
        prefix = start_prefix if start_prefix == end_prefix else start_prefix
        lo, hi = (start, end) if start <= end else (end, start)
        if hi - lo > 500:
            continue
        for seq in range(lo, hi + 1):
            add(prefix, seq)
        working = working.replace(match.group(0), " ")

    last_prefix: str | None = None
    tokens = re.split(r"\s*(?:/|,|;|\band\b)\s*", working)
    for token in tokens:
        token = token.strip().strip("-").strip()
        if not token:
            continue
        full = FULL_ID_RE.match(token)
        if full and not is_model_id(full.group(1).upper(), full.group(2), token[full.end() :]):
            last_prefix = full.group(1).upper()
            add(last_prefix, int(full.group(2)))
            continue
        bare = re.match(r"^(\d+)\b", token)
        if last_prefix and bare:
            add(last_prefix, int(bare.group(1)))
    if not found:
        for match in FULL_ID_RE.finditer(text):
            rest = text[match.end() : match.end() + 1]
            if is_model_id(match.group(1).upper(), match.group(2), rest):
                continue
            add(match.group(1).upper(), int(match.group(2)))
    return found


def parse_timestamp(raw: str | None) -> str | None:
    if not raw:
        return None
    text = raw.strip().strip("`").strip()
    text = re.sub(r"\s+\(.*\)$", "", text)
    if not text or text.lower() in {"not initiated", "-", "n/a"}:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    iso = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
    except ValueError:
        pass
    m = re.match(
        r"(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}(?::\d{2})?)(?:\.\d+)?)?(?:\s+([A-Z]{2,4}|[+-]\d{2}:?\d{2}))?",
        text,
    )
    if not m:
        return None
    date_part, time_part, tz_part = m.group(1), m.group(2), m.group(3)
    if not time_part:
        return date_part
    if time_part.count(":") == 1:
        time_part += ":00"
    naive = datetime.fromisoformat(f"{date_part}T{time_part}")
    if tz_part is None:
        tzinfo = UTC
    elif tz_part in TZ_OFFSETS:
        tzinfo = TZ_OFFSETS[tz_part]
    elif re.fullmatch(r"[+-]\d{4}", tz_part):
        sign = 1 if tz_part[0] == "+" else -1
        tzinfo = timezone(timedelta(hours=sign * int(tz_part[1:3]), minutes=sign * int(tz_part[3:5])))
    elif re.fullmatch(r"[+-]\d{2}:\d{2}", tz_part):
        sign = 1 if tz_part[0] == "+" else -1
        hours, minutes = tz_part[1:].split(":")
        tzinfo = timezone(timedelta(hours=sign * int(hours), minutes=sign * int(minutes)))
    else:
        tzinfo = UTC
    return naive.replace(tzinfo=tzinfo).astimezone(UTC).isoformat().replace("+00:00", "Z")


def extract_labeled_timestamp(body: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        match = re.search(
            rf"(?im)^(?:\s*[-*]\s*)?{re.escape(label)}\s*:\s*(.+)$",
            body,
        )
        if match:
            parsed = parse_timestamp(match.group(1))
            if parsed:
                return parsed
    return None


def file_date(path: Path) -> str | None:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
    return match.group(1) if match else None


def heading_is_item(title: str) -> bool:
    stripped = title.strip()
    if SKIP_HEADING_RE.match(stripped):
        return False
    if expand_ids(stripped):
        return True
    if re.match(r"^\d{4}-\d{2}-\d{2}\s+\S+", stripped) and not DATE_HEADING_RE.match(f"## {stripped}"):
        return True
    return False


def split_body_sections(body: str) -> dict[str, str]:
    buckets = {
        "evidence": [],
        "worknotes": [],
        "follow_ups": [],
        "acceptance": [],
        "hazards": [],
        "references": [],
        "goal": [],
    }
    current = "worknotes"
    mapping = {
        "evidence": "evidence",
        "verification": "evidence",
        "acceptance": "acceptance",
        "acceptance evidence": "acceptance",
        "result": "evidence",
        "goal": "goal",
        "follow-up": "follow_ups",
        "follow-ups": "follow_ups",
        "follow up": "follow_ups",
        "residual": "follow_ups",
        "residual scope": "follow_ups",
        "residual risk": "follow_ups",
        "hazard": "hazards",
        "hazards": "hazards",
        "known hazards": "hazards",
        "reference": "references",
        "references": "references",
        "changed tracked paths": "references",
        "changed paths": "references",
    }
    for line in body.splitlines():
        label = re.match(r"^\s{0,3}(?:#{2,3}\s+)?(?:[-*]\s+)?([A-Za-z][A-Za-z /_-]{0,40})\s*:\s*(.*)$", line)
        if label:
            key = label.group(1).strip().lower()
            current = mapping.get(key, current)
        buckets[current].append(line)
    return {k: "\n".join(v).strip() for k, v in buckets.items()}


def chunk_text(text: str, limit: int = 12000) -> list[str]:
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + limit])
        start += limit
    return chunks


def detect_status(title: str, body: str) -> str:
    for line in (title + "\n" + body).splitlines():
        if re.search(r"disposition\s*:", line, re.I):
            if re.search(r"\b(withdrawn|abandoned|cancelled|disposed-as-discarded|disposed)\b", line, re.I):
                if re.search(r"\bcompleted\b", line, re.I) and not re.search(
                    r"\b(withdrawn|abandoned|cancelled|disposed)\b", line, re.I
                ):
                    return "completed"
                return "abandoned"
    if re.search(r"\b(withdrawn|abandoned)\b", title, re.I):
        return "abandoned"
    return "completed"


def extract_status_line(body: str) -> str:
    """Return the Status field, joining a wrapped continuation line."""
    match = re.search(
        r"(?im)^status:\s*(.+?)(?:\n(?![A-Z][A-Za-z].*:)(.+?))?(?:\n|\Z)",
        body,
    )
    if not match:
        return ""
    first = match.group(1).strip()
    cont = (match.group(2) or "").strip()
    if cont and re.match(r"^(on|and|or|then|still)\b", cont, re.I):
        return f"{first} {cont}".strip()
    if cont and first.rstrip().endswith(("-", "gated", "on")):
        return f"{first} {cont}".strip()
    return first


def map_queue_status(title: str, body: str) -> str:
    """Map active-queue Status text onto an aawmdt lifecycle status."""
    status_line = extract_status_line(body)
    blob = f"{status_line}\n{title}\n{body}"
    lowered = blob.lower()
    if re.search(
        r"\b(withdrawn|abandoned|cancelled|disposed-as-discarded)\b",
        status_line,
        re.I,
    ):
        return "abandoned"
    if re.search(r"\bdeferred\b", status_line, re.I):
        return "deferred"
    if re.search(
        r"\b(dependency-gated|merge-gated|gated|blocked)\b",
        status_line,
        re.I,
    ):
        return "blocked"
    if re.search(r"\bin progress\b", status_line, re.I):
        return "in_progress"
    if re.search(r"\b(not initiated|queued)\b", status_line, re.I):
        return "open"
    if re.search(r"\bin progress\b", lowered):
        return "in_progress"
    if re.search(r"\b(gated|blocked)\b", lowered) and re.search(
        r"\b(status|gated on|blocked on|merge-gated)\b",
        lowered,
    ):
        return "blocked"
    return "open"


def _masked_dependency_text(body: str) -> str:
    """Drop inverted/downstream phrases so they are not treated as Depends-on."""
    masked = re.sub(
        r"root dependency for\s+.{0,200}",
        " ",
        body,
        flags=re.I | re.S,
    )
    masked = re.sub(r"\bfeeds\s+.{0,200}", " ", masked, flags=re.I | re.S)
    masked = re.sub(r"\bprecedes\s+.{0,200}", " ", masked, flags=re.I | re.S)
    return masked


def explicit_dependencies(body: str, self_ids: set[str], known_ids: set[str]) -> list[str]:
    deps: list[str] = []
    search_body = _masked_dependency_text(body)
    status_blob = extract_status_line(body)
    if status_blob:
        search_body = f"{status_blob}\n{search_body}"
    patterns = (
        r"(?:depends on|depended on|dependency-gated(?:\s+on)?|merge-gated on|merge gated on|"
        r"gated on|blocked on):?\s*"
        r"(`?[A-Z0-9][A-Z0-9, /.`'-]{0,120})",
        r"`([A-Z][A-Z0-9-]*-\d+)`\s+(?:is|was)\s+a\s+prereq",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, search_body, re.I):
            for prefix, seq, source_id in expand_ids(match.group(1)):
                stored = format_stored_id(prefix, seq, widths={})
                if stored in self_ids or source_id in self_ids:
                    continue
                deps.append(stored)
    for match in re.finditer(
        r"(?:prerequisite|depends on|blocked only on):?\s*((?:[A-Z][A-Z0-9-]*-\d+(?:\s*(?:/|,|and)\s*)?)+)",
        search_body,
        re.I,
    ):
        for prefix, seq, source_id in expand_ids(match.group(0)):
            deps.append(source_id)
    remaining_gate = re.search(
        r"dependency-gated(?:\s+on)?\s+(`?[A-Z0-9][A-Z0-9, /.`'-]{0,120})",
        status_blob,
        re.I,
    )
    if remaining_gate:
        for prefix, seq, source_id in expand_ids(remaining_gate.group(1)):
            stored = format_stored_id(prefix, seq, widths={})
            if stored in self_ids or source_id in self_ids:
                continue
            deps.append(stored)
    elif re.search(r"\b(gated|blocked|merge-gated|dependency-gated)\b", status_blob, re.I):
        for prefix, seq, source_id in expand_ids(status_blob):
            stored = format_stored_id(prefix, seq, widths={})
            if stored in self_ids or source_id in self_ids:
                continue
            deps.append(stored)
    unique = []
    seen = set()
    for source_id in deps:
        key = canonical_key_from_source(source_id)
        if key in seen or key in self_ids:
            continue
        if known_ids and key not in known_ids:
            continue
        seen.add(key)
        unique.append(source_id)
    return unique


def canonical_key_from_source(source_id: str) -> str:
    match = FULL_ID_RE.fullmatch(source_id)
    if not match:
        return source_id
    sequence = int(match.group(2))
    return f"{cli_prefix(match.group(1), sequence)}-{sequence}"


def format_stored_id(raw_prefix: str, sequence: int, widths: dict[str, int]) -> str:
    prefix = cli_prefix(raw_prefix, sequence)
    width = widths.get(prefix, max(3, len(str(sequence))))
    return f"{prefix}-{sequence:0{width}d}"


class Item:
    def __init__(
        self,
        path: Path,
        start: int,
        end: int,
        heading: str,
        body: str,
        ids: list[tuple[str, int, str]],
        kind: str,
    ) -> None:
        self.path = path
        self.start = start
        self.end = end
        self.heading = heading
        self.body = body
        self.ids = ids
        self.kind = kind

    @property
    def text(self) -> str:
        return (self.heading + "\n" + self.body).strip()


def _consume_bullet_item(lines: list[str], start: int) -> tuple[int, Item | None]:
    line = lines[start]
    bullet = BULLET_ITEM_RE.match(line)
    anon = ANON_BULLET_RE.match(line)
    if not ((bullet and len(bullet.group(1)) == 0) or anon):
        return start + 1, None
    i = start + 1
    n = len(lines)
    while i < n:
        nxt = lines[i]
        if HEADING_RE.match(nxt):
            break
        nxt_id = BULLET_ITEM_RE.match(nxt)
        if nxt_id and len(nxt_id.group(1)) == 0:
            break
        if ANON_BULLET_RE.match(nxt):
            break
        i += 1
    heading_text = line.lstrip("- ").strip()
    if bullet:
        ids = expand_ids(heading_id_region(heading_text)) or expand_ids(
            f"{bullet.group(2)}-{bullet.group(3)}"
        )
    else:
        ids = expand_ids(heading_id_region(heading_text))
    body = "\n".join(lines[start:i])
    return i, Item(Path(), start + 1, i, heading_text, body, ids, "bullet")


def parse_file(path: Path) -> list[Item]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    items: list[Item] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        heading = HEADING_RE.match(line)
        if DATE_HEADING_RE.match(line):
            i += 1
            while i < n:
                if HEADING_RE.match(lines[i]):
                    break
                nxt, item = _consume_bullet_item(lines, i)
                if item is None:
                    i += 1
                    continue
                item.path = path
                items.append(item)
                i = nxt
            continue
        if heading:
            level, title = heading.group(1), heading.group(2).strip()
            if FILE_TITLE_RE.match(line) or (
                SKIP_HEADING_RE.match(title) and not expand_ids(title)
            ):
                i += 1
                continue
            if heading_is_item(title):
                start = i
                ids = expand_ids(heading_id_region(title)) or expand_ids(title)
                i += 1
                while i < n:
                    nxt = HEADING_RE.match(lines[i])
                    if not nxt:
                        i += 1
                        continue
                    nxt_title = nxt.group(2).strip()
                    nxt_level = nxt.group(1)
                    if DATE_HEADING_RE.match(lines[i]) or FILE_TITLE_RE.match(lines[i]):
                        break
                    if heading_is_item(nxt_title) and len(nxt_level) <= len(level):
                        break
                    if len(nxt_level) < len(level):
                        break
                    i += 1
                body = "\n".join(lines[start + 1 : i])
                items.append(Item(path, start + 1, i, title, body, ids, "heading"))
                continue
        nxt, item = _consume_bullet_item(lines, i)
        if item is not None:
            item.path = path
            items.append(item)
            i = nxt
            continue
        i += 1
    return items


def as_start(ts: str) -> str:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", ts):
        return f"{ts}T00:00:00Z"
    return ts


def as_end(ts: str) -> str:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", ts):
        return f"{ts}T23:59:59Z"
    return ts


def later_ts(a: str | None, b: str | None) -> str | None:
    if a is None:
        return b
    if b is None:
        return a
    if a <= b:
        return b
    return a


def earlier_ts(a: str | None, b: str | None) -> str | None:
    if a is None:
        return b
    if b is None:
        return a
    return a if a <= b else b


def annotation(at: str, body: str, refs: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {"at": at, "actor": ACTOR, "body": body, "refs": refs}
    return payload


def build_current_snapshot(
    stored_id: str,
    prefix: str,
    sequence: int,
    status: str,
    goal: str,
    created_on: str,
    initiated_on: str | None,
    updated_on: str,
    acceptance: list[str],
    references: list[str],
    dependencies: list[str],
    hazards: list[str],
    worknotes: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    follow_ups: list[dict[str, Any]],
) -> dict[str, Any]:
    history = [
        {
            "at": created_on,
            "from_status": None,
            "to_status": "open",
            "note": "created",
            "actor": ACTOR,
        }
    ]
    current = "open"
    if initiated_on and initiated_on >= created_on:
        history.append(
            {
                "at": initiated_on,
                "from_status": "open",
                "to_status": "in_progress",
                "note": "started",
                "actor": ACTOR,
            }
        )
        current = "in_progress"
    if status != current:
        at = updated_on
        if at < history[-1]["at"]:
            at = history[-1]["at"]
        note = f"converted from active markdown queue as {status}"
        if status in {"blocked", "deferred"}:
            note = f"queue status mapped to {status}"
        history.append(
            {
                "at": at,
                "from_status": current,
                "to_status": status,
                "note": note,
                "actor": ACTOR,
            }
        )
    revision = max(2, len(history) + max(0, len(worknotes) + len(evidence) + len(follow_ups) - 1))
    return {
        "schema_version": 1,
        "id": stored_id,
        "prefix": prefix,
        "sequence": sequence,
        "revision": revision,
        "goal": goal,
        "acceptance_criteria": acceptance,
        "status": status,
        "timestamps": {
            "created_on": created_on,
            "initiated_on": initiated_on,
            "updated_on": updated_on,
            "completed_on": None,
        },
        "references": references,
        "dependencies": dependencies,
        "hazards": hazards,
        "worknotes": worknotes,
        "evidence": evidence,
        "follow_ups": follow_ups,
        "history": history,
    }


def existing_todo_ids() -> dict[str, str]:
    found: dict[str, str] = {}
    current = ANALYSIS / "current.jsonl"
    if current.is_file():
        for line in current.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                found[obj["id"]] = obj.get("status", "")
    for path in ANALYSIS.glob("archive/*/*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                found.setdefault(obj["id"], obj.get("status", ""))
    return found


def load_prefix_widths() -> dict[str, int]:
    prefixes: dict[str, int] = {}
    config = REPO / ".aawm-devtools.toml"
    if not config.is_file():
        return prefixes
    in_table = False
    for line in config.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[todo.prefixes]":
            in_table = True
            continue
        if stripped.startswith("[") and in_table:
            break
        if in_table and "=" in stripped and not stripped.startswith("#"):
            name, value = stripped.split("=", 1)
            prefixes[name.strip()] = int(value.strip())
    return prefixes


def write_current_jsonl(snapshots: list[dict[str, Any]]) -> None:
    dest = ANALYSIS / "current.jsonl"
    merged: dict[str, dict[str, Any]] = {}
    if dest.is_file():
        for line in dest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                merged[obj["id"]] = obj
    for item in snapshots:
        previous = merged.get(item["id"])
        if previous is None:
            merged[item["id"]] = item
            continue
        deps = list(dict.fromkeys([*(previous.get("dependencies") or []), *(item.get("dependencies") or [])]))
        previous["dependencies"] = deps
        for field in ("worknotes", "evidence", "follow_ups"):
            existing_notes = list(previous.get(field) or [])
            incoming = list(item.get(field) or [])
            seen_bodies = {
                note.get("body") if isinstance(note, dict) else str(note) for note in existing_notes
            }
            for note in incoming:
                body = note.get("body") if isinstance(note, dict) else str(note)
                if body not in seen_bodies:
                    existing_notes.append(note)
                    seen_bodies.add(body)
            previous[field] = existing_notes
        if not previous.get("goal") and item.get("goal"):
            previous["goal"] = item["goal"]
        merged[item["id"]] = previous
    ordered = sorted(merged.values(), key=lambda item: (item["prefix"], item["sequence"], item["id"]))
    dest.write_text(
        "".join(
            json.dumps(item, separators=(",", ":"), sort_keys=True, ensure_ascii=False) + "\n"
            for item in ordered
        ),
        encoding="utf-8",
    )


def convert_active_queues(mode: str) -> int:
    files = queue_files()
    scratch = SCRATCH
    scratch.mkdir(parents=True, exist_ok=True)
    before_path = scratch / "dwindle-before.txt"
    if not before_path.exists():
        rows = [
            f"{path.relative_to(REPO)}\t{sum(1 for _ in path.open(encoding='utf-8', errors='replace'))}"
            for path in files
        ]
        before_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    parsed: list[Item] = []
    for path in files:
        parsed.extend(parse_file(path))

    existing = existing_todo_ids()
    widths = load_prefix_widths()
    id_items: dict[str, list[Item]] = defaultdict(list)
    ledger_items: list[Item] = []

    def is_historical_mention(item: Item) -> bool:
        heading = item.heading.lower()
        return bool(
            re.search(r"\bclosed on 20\d{2}-\d{2}-\d{2}\b", heading)
            or re.search(r"\bresolved meta oauth\b", heading)
            or re.search(r"\bremain completed\b", heading)
        )

    for item in parsed:
        if is_historical_mention(item):
            continue
        if item.ids:
            for raw_prefix, seq, source_id in item.ids:
                if raw_prefix.upper() in MODEL_PREFIXES:
                    continue
                key = f"{cli_prefix(raw_prefix, seq)}|{seq}|{source_id}"
                id_items[key].append(item)
        elif item.kind in {"bullet", "heading"} and item.heading.strip().startswith("[ ]"):
            ledger_items.append(item)
        elif item.kind == "bullet" and not item.ids:
            ledger_items.append(item)

    prefix_width_from_seq: dict[str, int] = dict(widths)
    for key in id_items:
        cli_p, seq_s, _source_id = key.split("|", 2)
        seq = int(seq_s)
        prefix_width_from_seq[cli_p] = max(prefix_width_from_seq.get(cli_p, 3), max(3, len(str(seq))))
        prefix_width_from_seq[cli_p] = min(8, prefix_width_from_seq[cli_p])
    if ledger_items:
        prefix_width_from_seq["LEDGER"] = max(prefix_width_from_seq.get("LEDGER", 3), 3)
    if "D1" in prefix_width_from_seq:
        prefix_width_from_seq["D1"] = 3
    if "D1DATE" in prefix_width_from_seq:
        prefix_width_from_seq["D1DATE"] = 8
    widths = prefix_width_from_seq

    source_map: list[dict[str, Any]] = []
    ranges_by_file: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    merged_ids: dict[str, dict[str, Any]] = {}

    def add_range(item: Item) -> None:
        ranges_by_file[item.path].append((item.start, item.end))

    for key, group in id_items.items():
        cli_p, seq_s, source_id = key.split("|", 2)
        seq = int(seq_s)
        stored_id = f"{cli_p}-{seq:0{widths.get(cli_p, max(3, len(str(seq))))}d}"
        if stored_id in existing:
            for item in group:
                terminal = existing[stored_id] in {"completed", "abandoned"}
                if not terminal:
                    add_range(item)
                skipped.append(
                    {
                        "source_file": str(item.path.relative_to(REPO)),
                        "heading": item.heading,
                        "source_id": source_id,
                        "todo_id": stored_id,
                        "status": existing[stored_id],
                        "action": "skipped-existing",
                    }
                )
                source_map.append(
                    {
                        "source_file": str(item.path.relative_to(REPO)),
                        "start_line": item.start,
                        "end_line": item.end,
                        "heading": item.heading,
                        "source_id": source_id,
                        "todo_id": stored_id,
                        "status": existing[stored_id],
                        "kind": "skipped-existing",
                    }
                )
            continue
        status = "open"
        created_on = None
        initiated_on = None
        worknotes: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        follow_ups: list[dict[str, Any]] = []
        acceptance: list[str] = []
        hazards: list[str] = []
        references: list[str] = []
        goal = ""
        for item in group:
            add_range(item)
            mapped = map_queue_status(item.heading, item.body)
            if mapped == "blocked":
                status = "blocked"
            elif mapped == "deferred" and status != "blocked":
                status = "deferred"
            elif mapped == "in_progress" and status == "open":
                status = "in_progress"
            created_on = earlier_ts(
                created_on,
                extract_labeled_timestamp(item.body, ("Created on", "Created")),
            )
            initiated_raw = extract_labeled_timestamp(item.body, ("Initiated on", "Initiated"))
            initiated_on = earlier_ts(initiated_on, initiated_raw)
            sections = split_body_sections(item.body)
            if not goal:
                goal = goal_from_item(item)
            rel = str(item.path.relative_to(REPO))
            references.append(f"Legacy Markdown source: {rel}#{item.heading[:80]}")
            if sections["acceptance"]:
                for line in sections["acceptance"].splitlines():
                    cleaned = re.sub(r"^\s*[-*]\s*", "", line).strip()
                    if cleaned:
                        acceptance.append(cleaned[:500])
            if sections["hazards"]:
                for line in sections["hazards"].splitlines():
                    cleaned = re.sub(r"^\s*[-*]\s*", "", line).strip()
                    if cleaned:
                        hazards.append(cleaned[:500])
            for ref in re.findall(r"`([^`]{3,200})`", item.body)[:20]:
                cleaned = ref.strip()
                if cleaned and cleaned not in references:
                    references.append(cleaned)
            at = created_on or file_date(item.path) or "2026-01-01"
            source_block = f"{item.heading}\n\n{item.body}".strip()
            for chunk in chunk_text(source_block):
                worknotes.append(annotation(at, chunk, [rel]))
            note_chunks = []
            for label in ("Current state", "Immediate next action", "Investigation update"):
                match = re.search(
                    rf"(?is){re.escape(label)}\s*:\s*(.+?)(?:\n\n|\n[A-Z][a-z].*?:|\Z)",
                    item.body,
                )
                if match:
                    note_chunks.append(f"{label}: {match.group(1).strip()}")
            if note_chunks:
                evidence.append(annotation(at, "\n\n".join(note_chunks), [rel]))
            if sections["follow_ups"]:
                for chunk in chunk_text(sections["follow_ups"]):
                    follow_ups.append(annotation(at, chunk, [rel]))
            source_map.append(
                {
                    "source_file": rel,
                    "start_line": item.start,
                    "end_line": item.end,
                    "heading": item.heading,
                    "source_id": source_id,
                    "todo_id": stored_id,
                    "status": mapped,
                    "kind": item.kind,
                }
            )
        created_on = as_start(created_on or file_date(group[0].path) or "2026-01-01")
        updated_on = created_on
        if initiated_on:
            initiated_on = as_start(initiated_on) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", initiated_on) else initiated_on
            if initiated_on < created_on:
                initiated_on = created_on
            updated_on = later_ts(updated_on, initiated_on) or updated_on
        if status in {"in_progress", "blocked"} and initiated_on is None:
            # initiated_on is optional; blocked/open from gated items may never have started
            pass
        if status == "in_progress" and initiated_on is None:
            initiated_on = created_on
        if status in {"blocked", "deferred"}:
            updated_on = later_ts(updated_on, created_on) or created_on
        for field in (worknotes, evidence, follow_ups):
            for note in field:
                stamp = note.get("at") or created_on
                if stamp < created_on:
                    note["at"] = created_on
                if stamp > updated_on:
                    updated_on = stamp
        if source_id != stored_id:
            references.insert(0, f"source_id={source_id}")
        raw_prefix = source_id.rsplit("-", 1)[0]
        if cli_prefix(raw_prefix, seq) != raw_prefix:
            references.insert(0, f"source_prefix={raw_prefix}")
        merged_ids[stored_id] = {
            "source_ids": [source_id],
            "cli_prefix": cli_p,
            "sequence": seq,
            "status": status,
            "goal": (goal or stored_id)[:8000],
            "created_on": created_on,
            "initiated_on": initiated_on,
            "updated_on": updated_on,
            "acceptance": [item for item in acceptance[:40] if item.strip()],
            "references": [item for item in dict.fromkeys(references) if item.strip()][:40],
            "hazards": [item for item in hazards[:20] if item.strip()],
            "worknotes": worknotes,
            "evidence": evidence,
            "follow_ups": follow_ups,
            "self_keys": {f"{cli_p}-{seq}", stored_id, source_id},
            "siblings": [],
            "dep_sources": [item.body for item in group] + [item.heading for item in group],
        }

    ledger_seq = max((rec["sequence"] for rec in merged_ids.values() if rec["cli_prefix"] == "LEDGER"), default=0)
    seen_unprefixed: dict[tuple[str, str], str] = {}
    for item in ledger_items:
        add_range(item)
        fingerprint = (str(item.path.relative_to(REPO)), re.sub(r"\s+", " ", item.heading.strip().lower()))
        if fingerprint in seen_unprefixed:
            continue
        ledger_seq += 1
        stored_id = f"LEDGER-{ledger_seq:0{widths.get('LEDGER', 3)}d}"
        while stored_id in existing or stored_id in merged_ids:
            ledger_seq += 1
            stored_id = f"LEDGER-{ledger_seq:0{widths.get('LEDGER', 3)}d}"
        seen_unprefixed[fingerprint] = stored_id
        created_on = as_start(
            extract_labeled_timestamp(item.body, ("Created on", "Created"))
            or file_date(item.path)
            or "2026-01-01"
        )
        rel = str(item.path.relative_to(REPO))
        worknotes = [annotation(created_on, chunk, [rel]) for chunk in chunk_text(f"{item.heading}\n\n{item.body}".strip())]
        merged_ids[stored_id] = {
            "source_ids": [],
            "cli_prefix": "LEDGER",
            "sequence": ledger_seq,
            "status": map_queue_status(item.heading, item.body),
            "goal": goal_from_item(item)[:8000],
            "created_on": created_on,
            "initiated_on": None,
            "updated_on": created_on,
            "acceptance": [],
            "references": [f"Legacy Markdown source: {rel}#{item.heading[:80]}"],
            "hazards": [],
            "worknotes": worknotes,
            "evidence": [annotation(created_on, f"Unprefixed queue item from {rel}", [rel])],
            "follow_ups": [],
            "self_keys": {stored_id},
            "siblings": [],
            "dep_sources": [item.body],
        }
        source_map.append(
            {
                "source_file": rel,
                "start_line": item.start,
                "end_line": item.end,
                "heading": item.heading,
                "source_id": None,
                "todo_id": stored_id,
                "status": merged_ids[stored_id]["status"],
                "kind": "unprefixed",
            }
        )

    heading_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in source_map:
        if row.get("kind") in {"skipped-existing", "unprefixed"}:
            continue
        heading_groups[(row["source_file"], row["heading"])].append(row["todo_id"])
    for ids in heading_groups.values():
        unique = list(dict.fromkeys(ids))
        if len(unique) < 2:
            continue
        for stored_id in unique:
            rec = merged_ids.get(stored_id)
            if rec is None:
                continue
            rec.setdefault("siblings", [])
            rec["siblings"].extend(other for other in unique if other != stored_id)

    known_keys = {f"{rec['cli_prefix']}-{rec['sequence']}" for rec in merged_ids.values()}
    known_keys.update(canonical_key_from_source(todo_id) for todo_id in existing)
    known_keys.update(existing)
    snapshots: dict[str, dict[str, Any]] = {}
    for stored_id, rec in merged_ids.items():
        dep_ids = []
        self_keys = rec["self_keys"]
        for body in rec["dep_sources"]:
            for source_dep in explicit_dependencies(body, self_keys, known_keys):
                parts = source_dep.rsplit("-", 1)
                if len(parts) != 2 or not parts[1].isdigit():
                    continue
                dep_seq = int(parts[1])
                dep_cli = cli_prefix(parts[0], dep_seq)
                if dep_cli not in widths:
                    continue
                dep_stored = f"{dep_cli}-{dep_seq:0{widths[dep_cli]}d}"
                if dep_stored == stored_id:
                    continue
                if dep_stored not in merged_ids and dep_stored not in existing:
                    continue
                dep_ids.append(dep_stored)
        rec["dependencies"] = list(dict.fromkeys([*dep_ids, *(rec.get("siblings") or [])]))
        snapshots[stored_id] = build_current_snapshot(
            stored_id=stored_id,
            prefix=rec["cli_prefix"],
            sequence=rec["sequence"],
            status=rec["status"],
            goal=rec["goal"],
            created_on=rec["created_on"],
            initiated_on=rec["initiated_on"],
            updated_on=rec["updated_on"],
            acceptance=rec["acceptance"]
            or [f"Converted from active markdown queue for {stored_id}"],
            references=[item for item in rec["references"] if item and item.strip()],
            dependencies=[item for item in rec["dependencies"] if item and item.strip()],
            hazards=[item for item in rec["hazards"] if item and item.strip()],
            worknotes=rec["worknotes"],
            evidence=rec["evidence"],
            follow_ups=rec["follow_ups"],
        )

    inventory = {
        "file_count": len(files),
        "parsed_item_blocks": len(parsed),
        "unique_todo_ids": len(snapshots),
        "skipped_existing": len(skipped),
        "prefixes": widths,
        "status_counts": {
            status: sum(1 for item in snapshots.values() if item["status"] == status)
            for status in ("open", "in_progress", "blocked", "deferred")
        },
    }
    (scratch / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    (scratch / "conversion-map.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_map),
        encoding="utf-8",
    )
    print(json.dumps(inventory, indent=2))
    if mode == "inventory":
        return 0

    write_toml(widths)
    write_current_jsonl(list(snapshots.values()))
    init = run_aawmdt("init", json_out=True)
    (scratch / "todo-init.json").write_text(init.stdout or init.stderr, encoding="utf-8")
    if init.returncode != 0:
        print(init.stdout)
        print(init.stderr, file=sys.stderr)
        return init.returncode
    check = run_aawmdt("check", json_out=True)
    (scratch / "todo-check.json").write_text(check.stdout or check.stderr, encoding="utf-8")
    if check.returncode != 0:
        print(check.stdout)
        print(check.stderr, file=sys.stderr)
        return check.returncode

    if mode != "pilot":
        for path, ranges in ranges_by_file.items():
            merged_ranges = sorted(ranges)
            compact: list[tuple[int, int]] = []
            for start, end in merged_ranges:
                if compact and start <= compact[-1][1]:
                    compact[-1] = (compact[-1][0], max(compact[-1][1], end))
                else:
                    compact.append((start, end))
            strip_ranges(path, compact)
            if path.name.endswith(".todo.md") or path.name == "todo.md":
                text = path.read_text(encoding="utf-8")
                if not FULL_ID_RE.search(text) and "- [ ]" not in text:
                    path.write_text(
                        f"# {path.stem.replace('.', ' ').replace('todo', 'Queue').strip().title()}\n\n"
                        "No active items. Converted discrete work items to current `aawmdt todo` records.\n",
                        encoding="utf-8",
                    )

    after_rows = [
        f"{path.relative_to(REPO)}\t{sum(1 for _ in path.open(encoding='utf-8', errors='replace'))}"
        for path in files
    ]
    (scratch / "dwindle-after.txt").write_text("\n".join(after_rows) + "\n", encoding="utf-8")
    print(f"converted={len(snapshots)} skipped={len(skipped)} check_rc={check.returncode}")
    return 0


def build_snapshot(
    stored_id: str,
    prefix: str,
    sequence: int,
    status: str,
    goal: str,
    created_on: str,
    initiated_on: str | None,
    completed_on: str,
    updated_on: str,
    acceptance: list[str],
    references: list[str],
    dependencies: list[str],
    hazards: list[str],
    worknotes: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    follow_ups: list[dict[str, Any]],
) -> dict[str, Any]:
    history = [
        {
            "at": created_on,
            "from_status": None,
            "to_status": "open",
            "note": "created",
            "actor": ACTOR,
        }
    ]
    if initiated_on:
        history.append(
            {
                "at": initiated_on,
                "from_status": "open",
                "to_status": "in_progress",
                "note": "started",
                "actor": ACTOR,
            }
        )
        from_status = "in_progress"
        complete_at = later_ts(initiated_on, completed_on) or completed_on
    else:
        from_status = "open"
        complete_at = completed_on
    note = "converted from completed markdown ledger"
    if status == "abandoned":
        note = "withdrawn, abandoned, cancelled, or disposed in completed markdown ledger"
    history.append(
        {
            "at": complete_at,
            "from_status": from_status,
            "to_status": status,
            "note": note,
            "actor": ACTOR,
        }
    )
    revision = 1 + (1 if initiated_on else 0) + 1 + max(0, len(worknotes) + len(evidence) + len(follow_ups) - 1)
    return {
        "schema_version": 1,
        "id": stored_id,
        "prefix": prefix,
        "sequence": sequence,
        "revision": max(revision, 2),
        "goal": goal,
        "acceptance_criteria": acceptance,
        "status": status,
        "timestamps": {
            "created_on": created_on,
            "initiated_on": initiated_on,
            "updated_on": updated_on,
            "completed_on": completed_on,
        },
        "references": references,
        "dependencies": dependencies,
        "hazards": hazards,
        "worknotes": worknotes,
        "evidence": evidence,
        "follow_ups": follow_ups,
        "history": history,
    }


def run_aawmdt(*args: str, json_out: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = [AAWMDT, "todo", *args, "--root", str(REPO)]
    if json_out and "--json" not in args:
        cmd.append("--json")
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def write_toml(prefixes: dict[str, int]) -> None:
    lines = [
        "[todo]",
        'root = ".analysis"',
        'current_filename = "current.jsonl"',
        'manifest_filename = "todo-manifest.json"',
        'agent_event_filename = "todo-agent-events.jsonl"',
        'archive_directory_pattern = "archive/{year}"',
        'archive_filename_pattern = "{prefix}-{year}.jsonl"',
        'archive_timezone = "UTC"',
        "",
        "[todo.prefixes]",
    ]
    for prefix in sorted(prefixes):
        lines.append(f"{prefix} = {prefixes[prefix]}")
    (REPO / ".aawm-devtools.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def archive_year(completed_on: str) -> str:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", completed_on):
        return completed_on[:4]
    return completed_on[:4]


def strip_ranges(path: Path, ranges: list[tuple[int, int]]) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    remove = set()
    for start, end in ranges:
        for idx in range(start - 1, end):
            if 0 <= idx < len(lines):
                remove.add(idx)
    kept = [line for i, line in enumerate(lines) if i not in remove]
    text = "".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines_out = []
    for line in text.splitlines():
        if DATE_HEADING_RE.match(line) or FILE_TITLE_RE.match(line) or (
            HEADING_RE.match(line) and SKIP_HEADING_RE.match(HEADING_RE.match(line).group(2).strip())
        ):
            continue
        if not line.strip():
            continue
        lines_out.append(line)
    if not lines_out:
        stem = path.name
        text = (
            f"# Completed ledger\n\n"
            f"All discrete items from `{stem}` were converted to archived `aawmdt todo` records.\n"
        )
    else:
        title = f"# Completed ledger — {path.name}\n"
        remainder = "\n".join(lines_out).strip()
        text = title + "\n" + remainder + "\n"
        if not FULL_ID_RE.search(remainder) and len(remainder.splitlines()) <= 8:
            text = (
                f"# Completed ledger\n\n"
                f"All discrete items from `{path.name}` were converted to archived `aawmdt todo` records.\n"
            )
    path.write_text(text, encoding="utf-8")


def goal_title(heading: str) -> str:
    stripped = re.sub(r"^\d{4}-\d{2}-\d{2}\s+", "", heading.strip())
    region = heading_id_region(heading)
    ids = expand_ids(region)
    if not ids:
        return stripped or heading
    rest = stripped[len(region) :].lstrip(" \t-—:")
    return rest or stripped


def goal_from_item(item: Item) -> str:
    sections = split_body_sections(item.body)
    goal = sections.get("goal") or ""
    first = goal_title(item.heading)
    if not first:
        first = item.heading
    if goal:
        goal_line = re.sub(r"(?is)^.*?goal\s*:\s*", "", goal).strip()
        if goal_line:
            return f"{first}\n\n{goal_line[:4000]}"
    return first or item.heading


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "all"
    if mode in {"queues", "active", "inventory-queues"}:
        queue_mode = "inventory" if mode == "inventory-queues" else "all"
        return convert_active_queues(queue_mode)
    files = ledger_files()
    before_path = SCRATCH / "dwindle-before.txt"
    if mode in {"inventory", "all"} and not before_path.exists():
        rows = []
        for path in files:
            rows.append(f"{path.relative_to(REPO)}\t{sum(1 for _ in path.open(encoding='utf-8', errors='replace'))}")
        before_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    parsed: list[Item] = []
    for path in files:
        parsed.extend(parse_file(path))

    id_items: dict[str, list[Item]] = defaultdict(list)
    ledger_items: list[Item] = []
    for item in parsed:
        if item.ids:
            for raw_prefix, seq, source_id in item.ids:
                key = f"{cli_prefix(raw_prefix, seq)}|{seq}|{source_id}"
                id_items[key].append(item)
        else:
            ledger_items.append(item)

    prefix_max: dict[str, int] = defaultdict(int)
    prefix_width_from_seq: dict[str, int] = defaultdict(lambda: 3)
    outlier_keys = []
    for key in id_items:
        cli_p, seq_s, source_id = key.split("|", 2)
        seq = int(seq_s)
        if len(str(seq)) > 8:
            outlier_keys.append(key)
            continue
        prefix_max[cli_p] = max(prefix_max[cli_p], seq)
        prefix_width_from_seq[cli_p] = max(prefix_width_from_seq[cli_p], max(3, len(str(seq))))

    for key in outlier_keys:
        cli_p, seq_s, source_id = key.split("|", 2)
        mapped = "LEDGER"
        prefix_max[mapped] = prefix_max[mapped]
        prefix_width_from_seq[mapped] = 3

    if ledger_items:
        prefix_max["LEDGER"] = max(prefix_max.get("LEDGER", 0), len(ledger_items))
        prefix_width_from_seq["LEDGER"] = max(3, len(str(max(prefix_max["LEDGER"], 1))))

    widths = {p: min(8, prefix_width_from_seq[p]) for p in prefix_width_from_seq}
    if "LEDGER" in widths:
        widths["LEDGER"] = max(widths["LEDGER"], 3)

    known_stored: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    source_map: list[dict[str, Any]] = []
    ranges_by_file: dict[Path, list[tuple[int, int]]] = defaultdict(list)

    def add_range(item: Item) -> None:
        ranges_by_file[item.path].append((item.start, item.end))

    merged_ids: dict[str, dict[str, Any]] = {}
    for key, group in id_items.items():
        cli_p, seq_s, source_id = key.split("|", 2)
        seq = int(seq_s)
        if len(str(seq)) > 8:
            continue
        stored_id = f"{cli_p}-{seq:0{widths[cli_p]}d}"
        status = "completed"
        created_on = None
        initiated_on = None
        completed_on = None
        worknotes: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        follow_ups: list[dict[str, Any]] = []
        acceptance: list[str] = []
        hazards: list[str] = []
        references: list[str] = []
        goal = ""
        for item in group:
            add_range(item)
            item_status = detect_status(item.heading, item.body)
            if item_status == "abandoned":
                status = "abandoned"
            created_on = earlier_ts(
                created_on,
                extract_labeled_timestamp(item.body, ("Created on", "Created")),
            )
            initiated_on = earlier_ts(
                initiated_on,
                extract_labeled_timestamp(item.body, ("Initiated on", "Initiated")),
            )
            completed_on = later_ts(
                completed_on,
                extract_labeled_timestamp(
                    item.body,
                    ("Completed on", "Completed", "Live closeout on", "Source delivered on"),
                ),
            )
            if not completed_on:
                completed_on = file_date(item.path)
            sections = split_body_sections(item.body)
            if not goal:
                goal = goal_from_item(item)
            rel = str(item.path.relative_to(REPO))
            references.append(f"Legacy Markdown source: {rel}#{item.heading[:80]}")
            if sections["acceptance"]:
                for line in sections["acceptance"].splitlines():
                    cleaned = re.sub(r"^\s*[-*]\s*", "", line).strip()
                    if cleaned:
                        acceptance.append(cleaned[:500])
            if sections["hazards"]:
                for line in sections["hazards"].splitlines():
                    cleaned = re.sub(r"^\s*[-*]\s*", "", line).strip()
                    if cleaned:
                        hazards.append(cleaned[:500])
            refs_extra = re.findall(r"`([^`]{3,200})`", item.body)
            for ref in refs_extra[:20]:
                cleaned = ref.strip()
                if cleaned and cleaned not in references:
                    references.append(cleaned)
            at = completed_on or created_on or file_date(item.path) or "2026-01-01"
            source_block = f"{item.heading}\n\n{item.body}".strip()
            for idx, chunk in enumerate(chunk_text(source_block)):
                worknotes.append(annotation(at, chunk, [rel]))
            if sections["evidence"]:
                for chunk in chunk_text(sections["evidence"]):
                    evidence.append(annotation(at, chunk, [rel]))
            else:
                evidence.append(annotation(at, f"Closeout recorded in {rel}", [rel]))
            if sections["follow_ups"]:
                for chunk in chunk_text(sections["follow_ups"]):
                    follow_ups.append(annotation(at, chunk, [rel]))
            source_map.append(
                {
                    "source_file": rel,
                    "start_line": item.start,
                    "end_line": item.end,
                    "heading": item.heading,
                    "source_id": source_id,
                    "todo_id": stored_id,
                    "status": item_status,
                    "kind": item.kind,
                }
            )
        created_on = as_start(created_on or completed_on or file_date(group[0].path) or "2026-01-01")
        completed_on = as_end(completed_on or created_on)
        if created_on > completed_on:
            created_on = as_start(completed_on[:10])
        if initiated_on:
            initiated_on = as_start(initiated_on) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", initiated_on) else initiated_on
            if initiated_on < created_on:
                initiated_on = created_on
            if initiated_on > completed_on:
                initiated_on = completed_on
        updated_on = later_ts(completed_on, created_on) or completed_on
        for field in (worknotes, evidence, follow_ups):
            for note in field:
                at = note.get("at")
                if at and at < created_on:
                    note["at"] = created_on
                if at and at > updated_on:
                    note["at"] = updated_on
        if source_id != stored_id:
            references.insert(0, f"source_id={source_id}")
        raw_prefix = source_id.rsplit("-", 1)[0]
        if cli_prefix(raw_prefix, seq) != raw_prefix:
            references.insert(0, f"source_prefix={raw_prefix}")
        merged_ids[stored_id] = {
            "source_ids": [source_id],
            "raw_prefix": raw_prefix,
            "cli_prefix": cli_p,
            "sequence": seq,
            "status": status,
            "goal": goal[:8000] or stored_id,
            "created_on": created_on,
            "initiated_on": initiated_on,
            "completed_on": completed_on,
            "updated_on": updated_on,
            "acceptance": [item for item in acceptance[:40] if item.strip()],
            "references": [item for item in dict.fromkeys(references) if item.strip()][:40],
            "hazards": [item for item in hazards[:20] if item.strip()],
            "worknotes": worknotes,
            "evidence": evidence,
            "follow_ups": follow_ups,
            "self_keys": {f"{cli_p}-{seq}"},
            "siblings": [],
            "dep_sources": [],
        }
        for item in group:
            merged_ids[stored_id]["dep_sources"].append(item.body)
        known_stored.add(stored_id)

    ledger_seq = 0
    for item in ledger_items:
        add_range(item)
        ledger_seq += 1
        stored_id = f"LEDGER-{ledger_seq:0{widths.get('LEDGER', 3)}d}"
        created_on = as_start(
            extract_labeled_timestamp(item.body, ("Created on", "Created"))
            or file_date(item.path)
            or "2026-01-01"
        )
        initiated_on = extract_labeled_timestamp(item.body, ("Initiated on", "Initiated"))
        completed_on = as_end(
            extract_labeled_timestamp(item.body, ("Completed on", "Completed"))
            or file_date(item.path)
            or created_on
        )
        if created_on > completed_on:
            created_on = as_start(completed_on[:10])
        if initiated_on:
            initiated_on = as_start(initiated_on) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", initiated_on) else initiated_on
            if not (created_on <= initiated_on <= completed_on):
                initiated_on = None
        updated_on = later_ts(completed_on, created_on) or completed_on
        rel = str(item.path.relative_to(REPO))
        at = completed_on
        worknotes = [annotation(at, chunk, [rel]) for chunk in chunk_text(f"{item.heading}\n\n{item.body}".strip())]
        for note in worknotes:
            stamp = note.get("at")
            if stamp and stamp < created_on:
                note["at"] = created_on
            if stamp and stamp > updated_on:
                note["at"] = updated_on
        merged_ids[stored_id] = {
            "source_ids": [],
            "raw_prefix": "LEDGER",
            "cli_prefix": "LEDGER",
            "sequence": ledger_seq,
            "status": detect_status(item.heading, item.body),
            "goal": goal_from_item(item)[:8000],
            "created_on": created_on,
            "initiated_on": initiated_on if initiated_on and created_on <= initiated_on <= completed_on else None,
            "completed_on": completed_on,
            "updated_on": updated_on,
            "acceptance": [],
            "references": [f"Legacy Markdown source: {rel}#{item.heading[:80]}"],
            "hazards": [],
            "worknotes": worknotes,
            "evidence": [annotation(at, f"Unprefixed completed-ledger item from {rel}", [rel])],
            "follow_ups": [],
            "self_keys": {stored_id},
            "siblings": [],
            "dep_sources": [item.body],
        }
        source_map.append(
            {
                "source_file": rel,
                "start_line": item.start,
                "end_line": item.end,
                "heading": item.heading,
                "source_id": None,
                "todo_id": stored_id,
                "status": merged_ids[stored_id]["status"],
                "kind": "unprefixed",
            }
        )
        known_stored.add(stored_id)

    heading_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in source_map:
        heading_groups[(row["source_file"], row["heading"])].append(row["todo_id"])
    for ids in heading_groups.values():
        unique = list(dict.fromkeys(ids))
        if len(unique) < 2:
            continue
        for stored_id in unique:
            rec = merged_ids.get(stored_id)
            if rec is None:
                continue
            rec.setdefault("siblings", [])
            rec["siblings"].extend(other for other in unique if other != stored_id)

    known_keys = set()
    for stored_id, rec in merged_ids.items():
        known_keys.add(f"{rec['cli_prefix']}-{rec['sequence']}")

    snapshots: dict[str, dict[str, Any]] = {}
    for stored_id, rec in merged_ids.items():
        dep_ids = []
        self_keys = rec["self_keys"]
        for body in rec["dep_sources"]:
            for source_dep in explicit_dependencies(body, self_keys, known_keys):
                key = canonical_key_from_source(source_dep)
                prefix, seq_s = key.split("-", 1)
                # key is PREFIX-seq without padding
                parts = source_dep.rsplit("-", 1)
                dep_seq = int(parts[1])
                dep_cli = cli_prefix(parts[0], dep_seq)
                if len(str(dep_seq)) > 8:
                    continue
                if dep_cli not in widths:
                    continue
                dep_stored = f"{dep_cli}-{dep_seq:0{widths[dep_cli]}d}"
                if dep_stored == stored_id or dep_stored not in merged_ids:
                    continue
                dep_ids.append(dep_stored)
        siblings = rec.get("siblings") or []
        rec["dependencies"] = list(dict.fromkeys([*dep_ids, *siblings]))
        snapshots[stored_id] = build_snapshot(
            stored_id=stored_id,
            prefix=rec["cli_prefix"],
            sequence=rec["sequence"],
            status=rec["status"],
            goal=rec["goal"],
            created_on=rec["created_on"],
            initiated_on=rec["initiated_on"],
            completed_on=rec["completed_on"],
            updated_on=rec["updated_on"],
            acceptance=rec["acceptance"] or [f"Closeout recorded in completed markdown for {stored_id}"],
            references=[item for item in rec["references"] if item and item.strip()],
            dependencies=[item for item in rec["dependencies"] if item and item.strip()],
            hazards=[item for item in rec["hazards"] if item and item.strip()],
            worknotes=rec["worknotes"],
            evidence=rec["evidence"],
            follow_ups=rec["follow_ups"],
        )

    inventory = {
        "file_count": len(files),
        "parsed_item_blocks": len(parsed),
        "unique_todo_ids": len(snapshots),
        "unprefixed_items": len(ledger_items),
        "outlier_skipped": outlier_keys,
        "prefixes": widths,
        "hyphen_prefix_map": HYPHEN_PREFIX_MAP,
        "status_counts": {
            "completed": sum(1 for s in snapshots.values() if s["status"] == "completed"),
            "abandoned": sum(1 for s in snapshots.values() if s["status"] == "abandoned"),
        },
    }
    (SCRATCH / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    (SCRATCH / "conversion-map.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_map),
        encoding="utf-8",
    )
    print(json.dumps(inventory, indent=2))
    if mode == "inventory":
        return 0

    write_toml(widths)

    by_archive: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for snapshot in snapshots.values():
        year = archive_year(snapshot["timestamps"]["completed_on"])
        prefix = snapshot["prefix"]
        dest = ANALYSIS / "archive" / year / f"{prefix}-{year}.jsonl"
        by_archive[dest].append(snapshot)

    archive_root = ANALYSIS / "archive"
    if archive_root.exists():
        for stale in archive_root.rglob("*.jsonl"):
            if stale not in by_archive:
                stale.unlink()
    for dest, rows in by_archive.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(rows, key=lambda item: (item["prefix"], item["sequence"], item["id"]))
        dest.write_text(
            "".join(
                json.dumps(item, separators=(",", ":"), sort_keys=True, ensure_ascii=False) + "\n"
                for item in ordered
            ),
            encoding="utf-8",
        )

    init = run_aawmdt("init", json_out=True)
    (SCRATCH / "todo-init.json").write_text(init.stdout or init.stderr, encoding="utf-8")
    if init.returncode != 0:
        print(init.stdout)
        print(init.stderr, file=sys.stderr)
        return init.returncode
    check = run_aawmdt("check", json_out=True)
    (SCRATCH / "todo-check.json").write_text(check.stdout or check.stderr, encoding="utf-8")
    if check.returncode != 0:
        print(check.stdout)
        print(check.stderr, file=sys.stderr)
        return check.returncode

    if mode != "pilot":
        for path, ranges in ranges_by_file.items():
            merged_ranges = sorted(ranges)
            compact: list[tuple[int, int]] = []
            for start, end in merged_ranges:
                if compact and start <= compact[-1][1]:
                    compact[-1] = (compact[-1][0], max(compact[-1][1], end))
                else:
                    compact.append((start, end))
            strip_ranges(path, compact)

    after_rows = []
    for path in files:
        after_rows.append(f"{path.relative_to(REPO)}\t{sum(1 for _ in path.open(encoding='utf-8', errors='replace'))}")
    (SCRATCH / "dwindle-after.txt").write_text("\n".join(after_rows) + "\n", encoding="utf-8")
    print(f"converted={len(snapshots)} check_rc={check.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
