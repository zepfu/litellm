import re
from pathlib import Path

from scripts import convert_completed_ledgers_to_aawmdt_todo as conv


def test_expand_ids_keeps_source_prefix_and_sequence() -> None:
    assert conv.expand_ids("D1-1234") == [("D1", 1234, "D1-1234")]
    assert conv.cli_prefix("D1", 1234) == "D1"
    assert conv.format_stored_id("D1", 1234, {"D1": 3}) == "D1-1234"


def test_expand_ids_range_and_combined_heading() -> None:
    ranged = conv.expand_ids("MUSE-001 through MUSE-010")
    assert [item[2] for item in ranged] == [f"MUSE-{i}" for i in range(1, 11)]
    combined = conv.expand_ids("OPENAI-011 / D1-627")
    assert [item[2] for item in combined] == ["OPENAI-11", "D1-627"]
    bare = conv.expand_ids("D1-551 / 524 / 550 / 610")
    assert [item[2] for item in bare] == ["D1-551", "D1-524", "D1-550", "D1-610"]


def test_cli_prefix_maps_hyphen_and_date_sequences() -> None:
    assert conv.cli_prefix("ALI-QUOTA") == "ALIQUOTA"
    assert conv.cli_prefix("P-INV") == "PINV"
    assert conv.cli_prefix("D1", 20260602) == "D1DATE"
    assert conv.format_stored_id("D1", 20260602, {"D1DATE": 8}) == "D1DATE-20260602"


def test_heading_id_region_ignores_foreign_ids_in_title() -> None:
    title = "2026-06-16 D1-284 - Triage TAP CD-181 proxy security"
    region = conv.heading_id_region(title)
    assert conv.expand_ids(region) == [("D1", 284, "D1-284")]


def test_explicit_dependencies_allows_optional_colon() -> None:
    deps = conv.explicit_dependencies(
        "Depends on: OPENAI-008.",
        self_ids={"OPENAI-9", "OPENAI-009"},
        known_ids={"OPENAI-8", "OPENAI-9"},
    )
    assert deps == ["OPENAI-008"]
    without_colon = conv.explicit_dependencies(
        "Depends on OPENAI-008",
        self_ids=set(),
        known_ids={"OPENAI-8"},
    )
    assert without_colon == ["OPENAI-008"]


def test_map_queue_status_in_progress_open_and_gated() -> None:
    assert conv.map_queue_status("CURSOR-016", "Status: In progress. Oracle rank 1") == "in_progress"
    assert conv.map_queue_status("D1-766", "Status: Not initiated.") == "open"
    assert conv.map_queue_status("OPENAI-055", "Status: Queued for compatibility discovery") == "open"
    assert conv.map_queue_status("D1-611", "Status: Gated. `D1-610` source landed") == "blocked"
    assert (
        conv.map_queue_status("OPENAI-052", "Status: Dependency-gated on `OPENAI-039` and `OPENAI-040`")
        == "blocked"
    )


def test_explicit_dependencies_gated_on_and_merge_gated() -> None:
    gated = conv.explicit_dependencies(
        "Status: Dependency-gated on `OPENAI-039` and `OPENAI-040`.",
        self_ids=set(),
        known_ids={"OPENAI-39", "OPENAI-40", "OPENAI-52"},
    )
    assert "OPENAI-039" in gated
    assert "OPENAI-040" in gated
    merge_gated = conv.explicit_dependencies(
        "OPENAI-052 remains merge-gated on OPENAI-040.",
        self_ids={"OPENAI-52", "OPENAI-052"},
        known_ids={"OPENAI-40", "OPENAI-52"},
    )
    assert merge_gated == ["OPENAI-040"]
    colon = conv.explicit_dependencies(
        "Depends on: OPENAI-008.",
        self_ids=set(),
        known_ids={"OPENAI-8"},
    )
    assert colon == ["OPENAI-008"]
    gated = conv.explicit_dependencies(
        "Status: Gated. `D1-610` source landed as `2ce2fffb57` on `origin/develop`.",
        self_ids={"D1-611"},
        known_ids={"D1-610", "D1-611"},
    )
    assert gated == ["D1-610"]


def test_explicit_dependencies_joins_wrapped_dependency_gated_status() -> None:
    body = (
        "Status: `OPENAI-034/035/044` prerequisites delivered; still dependency-gated\n"
        "on `OPENAI-048`. Oracle global rank 13/20, High, confidence A (99%).\n"
    )
    deps = conv.explicit_dependencies(
        body,
        self_ids={"OPENAI-045", "OPENAI-45"},
        known_ids={"OPENAI-34", "OPENAI-35", "OPENAI-44", "OPENAI-45", "OPENAI-48"},
    )
    assert "OPENAI-048" in deps
    assert "OPENAI-034" not in deps
    assert "OPENAI-035" not in deps
    assert "OPENAI-044" not in deps


def test_root_dependency_for_does_not_invert_edge() -> None:
    body = (
        "Known hazards and dependencies: Root dependency for `ALI-013`, `ALI-015`,\n"
        "`ALI-016`, and `ALI-021` through `ALI-023`. Legacy rows without identity must\n"
        "fail to unknown rather than exhausted.\n"
    )
    deps = conv.explicit_dependencies(
        body,
        self_ids={"ALI-008", "ALI-8"},
        known_ids={"ALI-8", "ALI-13", "ALI-15", "ALI-16", "ALI-21", "ALI-23"},
    )
    assert deps == []
    feeds = conv.explicit_dependencies(
        "Known hazards and dependencies: Depends on `ALI-008`; feeds `ALI-013` and `ALI-016`.",
        self_ids={"ALI-015", "ALI-15"},
        known_ids={"ALI-8", "ALI-13", "ALI-15", "ALI-16"},
    )
    assert feeds == ["ALI-008"]


def test_fail_closed_heading_is_not_historical() -> None:
    item = conv.Item(
        Path("openrouter.todo.md"),
        1,
        2,
        "OR-021 - Fail closed on malformed native Responses success bodies",
        "Status: Not initiated.",
        [("OR", 21, "OR-21")],
        "heading",
    )
    heading = item.heading.lower()
    assert not re.search(r"\bclosed on 20\d{2}-\d{2}-\d{2}\b", heading)


def test_goal_title_keeps_unprefixed_leading_capital() -> None:
    heading = "2026-06-23 Prod promotion - LiteLLM aawm.90"
    assert conv.goal_title(heading) == "Prod promotion - LiteLLM aawm.90"
    item = conv.Item(
        Path("x.md"),
        1,
        2,
        heading,
        "Created on: 2026-06-23\nGoal: promote aawm.90",
        [],
        "heading",
    )
    goal = conv.goal_from_item(item)
    assert goal.startswith("Prod promotion")
