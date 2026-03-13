"""Entry point for the Prompt Fragment Analyzer.

Usage:
  python -m scripts.prompt_analyzer.cli analyze \\
      --captures ./captures/ \\
      --output ./replacement_table.json \\
      [--verbose]
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from typing import Optional

from scripts.prompt_analyzer.catalog import CatalogIndex
from scripts.prompt_analyzer.classifier import classify
from scripts.prompt_analyzer.differ import Differ
from scripts.prompt_analyzer.ingest import load_captures
from scripts.prompt_analyzer.models import (
    Fragment,
    Observation,
    ReplacementEntry,
    RequestSnapshot,
)
from scripts.prompt_analyzer.storage import JsonFileStorage

logger = logging.getLogger(__name__)


def _get_assistant_tool_calls(snapshot: Optional[RequestSnapshot]) -> list[str]:
    """Extract tool names used by the assistant in the last turn."""
    if snapshot is None:
        return []
    tool_calls: list[str] = []
    for msg in snapshot.messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "")
                    if name:
                        tool_calls.append(name)
    return tool_calls


def _build_replacement_entry(
    fragment: Fragment,
    request_id: str,
    existing: Optional[ReplacementEntry],
) -> ReplacementEntry:
    preview = fragment.text[:200]
    if existing is not None:
        return ReplacementEntry(
            fingerprint=fragment.fp,
            fragment_type=fragment.fragment_type,
            catalog_match=fragment.catalog_match,
            original_preview=existing.original_preview,
            replacement_text=existing.replacement_text,
            enabled=existing.enabled,
            observation_count=existing.observation_count + 1,
            first_seen=existing.first_seen,
            last_seen=request_id,
        )
    return ReplacementEntry(
        fingerprint=fragment.fp,
        fragment_type=fragment.fragment_type,
        catalog_match=fragment.catalog_match,
        original_preview=preview,
        replacement_text=None,
        enabled=False,
        observation_count=1,
        first_seen=request_id,
        last_seen=request_id,
    )


def _print_fragment(fragment: Fragment, request_id: str, verbose: bool) -> None:
    if not verbose:
        return
    preview = fragment.text[:120].replace("\n", "\\n")
    print(
        f"  [{fragment.source.value}] {fragment.fragment_type.value}"
        f" ({fragment.confidence:.2f}) trigger={fragment.trigger.value}"
        f"\n    preview: {preview!r}"
    )


def run_analyze(captures_dir: str, output_path: str, verbose: bool) -> int:
    """Main analysis pipeline. Returns exit code."""
    # Load captures
    snapshots = load_captures(captures_dir)
    if not snapshots:
        print("No captures found. Exiting.", file=sys.stderr)
        return 1

    print(f"Loaded {len(snapshots)} capture(s).")

    # Load (or create empty) replacement table
    storage = JsonFileStorage(output_path)
    entries = storage.load_entries()
    print(f"Existing replacement table: {len(entries)} entries.")

    # Catalog is a stub — returns empty results gracefully
    catalog = CatalogIndex({})

    # Pipeline
    differ = Differ()
    all_observations: list[Observation] = []
    type_counter: Counter = Counter()
    trigger_counter: Counter = Counter()

    for turn_index, curr in enumerate(snapshots):
        prev = snapshots[turn_index - 1] if turn_index > 0 else None
        is_first = turn_index == 0

        prev_tool_calls = _get_assistant_tool_calls(prev)
        new_fragments = differ.diff_requests(prev, curr)

        if verbose and new_fragments:
            print(f"\n[{curr.request_id}] agent={curr.agent} model={curr.model}")
            print(f"  {len(new_fragments)} new fragment(s):")

        for fragment in new_fragments:
            classified = classify(
                fragment,
                prev_assistant_tool_calls=prev_tool_calls,
                catalog=catalog,
                is_first_request=is_first,
            )

            obs = Observation(
                fragment=classified,
                request_id=curr.request_id,
                turn_index=turn_index,
                is_new=True,
                previous_tool_calls=prev_tool_calls,
            )
            all_observations.append(obs)

            existing = entries.get(classified.fp)
            entries[classified.fp] = _build_replacement_entry(
                classified, curr.request_id, existing
            )

            type_counter[classified.fragment_type.value] += 1
            trigger_counter[classified.trigger.value] += 1

            _print_fragment(classified, curr.request_id, verbose)

    # Persist
    storage.save_all(entries)
    print(f"\nReplacement table written to: {output_path}")
    print(f"Total entries: {len(entries)}")

    # Summary
    print("\n--- Fragment type distribution ---")
    for ftype, count in sorted(type_counter.items(), key=lambda x: -x[1]):
        print(f"  {ftype:25s} {count}")

    print("\n--- Trigger distribution ---")
    for trigger, count in sorted(trigger_counter.items(), key=lambda x: -x[1]):
        print(f"  {trigger:25s} {count}")

    total = sum(type_counter.values())
    print(f"\nTotal new fragments observed: {total}")

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prompt Fragment Analyzer — diff and classify Claude Code API captures."
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze capture files and build replacement table."
    )
    analyze_parser.add_argument(
        "--captures",
        default="./captures",
        help="Path to captures directory (default: ./captures)",
    )
    analyze_parser.add_argument(
        "--output",
        default="./replacement_table.json",
        help="Path to output JSON file (default: ./replacement_table.json)",
    )
    analyze_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each fragment as it is found.",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "analyze":
        return run_analyze(args.captures, args.output, args.verbose)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
