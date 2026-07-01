# AAWM model output replay (local harness)

Fixture-first local replay for malformed model or tool-call output. The harness
routes sanitized payloads through existing production detector, repair, intake,
and scorer helpers without live provider traffic or proxy hot-path changes.

## Safety

- Fixtures under `tests/fixtures/model_output_replay/` are **sanitized** only.
  `manifest.jsonl` records `redaction_status` per fixture.
- CLI and tests emit **compact classification/evidence** only. Raw transcript
  strings (for example lines containing `Tool label:` or `Input payload:`) are
  not printed by default.
- No provider credentials and no outbound model requests are required.

## Manifest schema

Each `manifest.jsonl` row requires:

- `fixture_id`, `source_kind`, `provider_family`, `transcript_surface`
- `redaction_status`, `source_signal`, `expected_disposition`
- `lanes`, `tool_names`, `sanitized_text_ref`
- optional `raw_source_ref`, `notes`

Fixture JSON files reference `response_body` (and optional `request_body`,
`intake_context`, adapter metadata) used by production helpers.

Use `source_kind`, `transcript_surface`, and `source_signal` to classify what the
fixture represents. Phase 1 fixtures should make it clear whether the evidence is
model-authored malformed output, adapter-visible text leakage, scorer-only
evidence, or orchestrator/session contamination; the CLI includes these fields
in compact output so reviewers can tell which class was exercised without seeing
the transcript body.

## Disposition

Replay is **fail-closed** unless an advertised-tool, schema-valid repair path
is proven:

- `detect` uses `_is_codex_auto_agent_malformed_tool_call_text_output` and
  `extract_malformed_tool_call_evidence`.
- `repair` uses
  `_try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body`
  with the fixture `request_body` tools list.
- `intake` builds records via `build_malformed_tool_call_intake_context` and
  `build_malformed_tool_call_intake_record`.
- `scorer` runs `score_agent_quality_context` on assistant text extracted from
  the response body.

## CLI usage

```bash
./.venv/bin/python scripts/dev-smoke/model_output_replay.py \
  --fixture grok_literal_exec_command_repairable \
  --lane all \
  --json
```

Arguments:

- `--manifest` (default: `tests/fixtures/model_output_replay/manifest.jsonl`)
- `--fixture` (required `fixture_id`)
- `--lane` (`detect`, `repair`, `intake`, `scorer`, comma-separated, or `all`)
- `--json` (machine-readable compact output)

## Tests

```bash
./.venv/bin/python -m pytest tests/test_scripts/test_model_output_replay.py -q \
  --tb=short -p no:rerunfailures --override-ini=addopts=
```

Programmatic access: `tests/support/model_output_replay.py` (`load_manifest`,
`resolve_fixture`, `run_replay`, `run_replay_lane`).
