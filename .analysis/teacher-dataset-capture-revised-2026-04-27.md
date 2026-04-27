# Teacher Dataset Capture Revised Recommendation

## Goal

Build a usable dataset for later fine-tuning or preference-training open source
models, without putting training-data decisions in the live LiteLLM request
path.

The immediate goal is capture, curation, annotation, and export. Actual tuning
is a separate downstream concern.

## Core Architecture

Keep the live proxy path as capture-only.

The live LiteLLM / AAWM path should persist durable facts:

- `public.session_history`
- `public.session_history_tool_activity`
- Langfuse trace / observation lineage
- provider, model, tenant, agent, client, environment, runtime version
- token, cost, cache, reasoning, permission-check, and tool activity facts
- enough request/response lineage to reconstruct the user-visible interaction

Do not put dataset selection, scoring, distillation, preference pairing, or
export shaping in the request path.

Those decisions belong in an offline dataset worker that can be rerun as labels,
schemas, and policies improve.

## Current Best Source Tables

Use `public.session_history` as the candidate index.

It already carries:

- `session_id`, `trace_id`, `litellm_call_id`
- provider/model/model group
- agent and tenant identity
- client/runtime identity
- environment/version/wheel metadata
- token and cost fields
- reasoning presence/source/counts
- provider-cache status/miss telemetry
- permission-check token/cost fields
- tool count and file/git rollups
- metadata containing source trace/observation details

Use `public.session_history_tool_activity` as the tool-behavior table.

It already carries:

- tool name/kind/index/call id
- tool arguments
- command text
- file reads and modifications
- git commit/push counts

Use Langfuse as the raw trace/payload source when the exporter needs original
input/output, intermediate observations, or transcript detail. Langfuse should
not be the canonical dataset store.

## Important Missing Capture

The main current gap for high-quality negative examples is durable tool
result/error capture.

We can see tool calls and arguments, but failure records are much more useful if
they include the paired tool result/error. Example:

- bad call: `Read({"file_path": "...", "pages": ""})`
- tool error: `Invalid pages parameter`
- retry pattern: repeated identical failed call
- corrected action: `Read({"file_path": "..."})`

If Langfuse transcripts reliably contain the tool result, the offline exporter
can reconstruct this. If not, extend `session_history_tool_activity` or add a
companion table with:

- `tool_result_text`
- `tool_error_text`
- `tool_status`
- `tool_latency_ms`
- optional result size/count metadata

Keep this as factual capture only. Do not classify it in the live path.

## Dataset Record Types

Store positives and negatives separately. Do not mix failed traces into the
imitation set.

Recommended record types:

- `positive_demonstration`: successful traces worth imitating.
- `negative_outcome`: failed traces useful as examples of what not to do.
- `repair_example`: failed step plus corrected next action.
- `preference_pair`: bad attempt paired with a better attempt for the same or
  similar task.
- `critique_example`: trace plus structured explanation of what went wrong.

## Positive Candidate Rules

Start with deterministic gates before any LLM review:

- successful final response
- no runtime blocker signatures
- no tool/request failure
- complete provider/model/session identity
- complete token/cost accounting
- expected tenant and environment
- provider/model allowlist
- no control-plane/probe/internal-check classification
- optionally, harness-passed session/case when available

Positive records should preserve:

- normalized input
- relevant preserved system/project context
- final answer
- concise plan/rationale summary, if derived
- checks performed
- tool sequence summary
- lineage back to source trace/session/observation

## Negative Candidate Rules

Negatives should be captured and labeled, but not trained as behavior to imitate.

Useful deterministic negative labels:

- `invalid_tool_args`
- `repeated_tool_error`
- `retry_loop`
- `provider_error`
- `rate_limited`
- `tool_unavailable`
- `validation_failed`
- `incomplete`
- `wrong_final_answer`
- `excessive_cost`
- `excessive_turns`
- `control_plane_noise`
- `permission_check_noise`

Specific example:

- `Read.pages=""` plus `Invalid pages parameter` should become
  `invalid_tool_args`.
- Repeated identical failed `Read` calls should additionally become
  `repeated_tool_error` / `retry_loop`.
- A useful repair target is the corrected call with `pages` omitted.

## Offline Worker Shape

Add a standalone process such as `scripts/export_teacher_dataset.py`.

Suggested stages:

1. Candidate query
   - Query `session_history` by date/session/provider/model/tenant/environment.
   - Join `session_history_tool_activity`.
   - Exclude known internal/probe/control-plane records by default.

2. Trace hydration
   - Fetch Langfuse trace/observation payloads only for selected candidates.
   - Keep source ids and hashes for lineage.

3. Deterministic classification
   - Apply rule-based labels for obvious positives and negatives.
   - Detect repeated failed tool calls, invalid tool arguments, provider errors,
     missing final output, wrong environment, and incomplete accounting.

4. LLM-assisted annotation
   - Use an LLM to produce structured annotations for candidates that pass
     deterministic prefilters.
   - The LLM should cite evidence from the trace/tool rows.
   - The LLM should not be the only authority for hard factual labels.

5. Validation
   - Enforce schema validation, allowed labels, lineage fields, hashes, and
     confidence thresholds.
   - Optionally route ambiguous or high-value examples to human review.

6. Storage/export
   - Write normalized records to an AAWM-managed dataset table/schema or JSONL
     export directory.
   - Support separate exports for SFT positives, preference pairs, repair
     examples, critiques, and eval sets.

## LLM Annotation Contract

The annotation model should emit structured JSON only.

Suggested fields:

- `record_type`
- `labels`
- `quality_score`
- `confidence`
- `failure_mode`
- `evidence`
- `corrected_next_action`
- `training_suitability`
- `redaction_required`
- `notes`

Evidence should reference concrete source locations such as:

- session id
- trace id
- observation id
- tool call id/index
- tool name
- provider/model
- error text snippets

The annotation prompt should explicitly distinguish:

- bad outcome worth storing
- bad outcome worth excluding
- positive demonstration
- repair candidate
- preference-pair candidate
- internal/control-plane noise

## Dataset Storage Fields

Recommended canonical fields:

- `dataset_record_id`
- `record_type`
- `labels`
- `source_session_id`
- `source_trace_id`
- `source_observation_ids`
- `source_litellm_call_ids`
- `teacher_provider`
- `teacher_model`
- `teacher_client`
- `tenant_id`
- `agent_name`
- `litellm_environment`
- `litellm_version`
- `prompt_policy_version`
- `input`
- `output`
- `tool_events`
- `failure_events`
- `derived_plan`
- `derived_checks`
- `critique`
- `repair_action`
- `quality_score`
- `confidence`
- `approval_status`
- `dataset_split`
- `content_hash`
- `lineage_hash`
- `created_at`

## Provider Notes

### Claude / Anthropic

Best initial positive teacher source because its traces are usually easier to
interpret. Do not train directly on raw hidden reasoning. Use it to derive
plans, checks, critiques, and repair labels.

### OpenAI / Codex

Useful for final answers, tool behavior, retries, repair examples, and
preference pairs. Recent session-history/tool-activity work makes Codex/OpenAI
tool-use traces much more useful than they were when the first teacher note was
written.

### Gemini

Useful for outcome, tool behavior, and correction patterns. Thought signatures
should remain correlation/metadata signals, not training text.

### OpenRouter / NVIDIA

Useful once provider/model/cost/session-history invariants are clean for the
specific lane. Free/canary/provider-unavailable cases are useful as negative or
reliability examples, not default positive demonstrations.

### Embedding / Rerank / Permission / Control Plane

Exclude from default teacher datasets. They may be useful for operational
analytics or separate eval datasets, but they are not general agent training
examples.

## Recommended MVP

1. Add an offline exporter that reads `session_history` and
   `session_history_tool_activity`.
2. Export positive Claude examples from harness-passed or clearly successful
   sessions.
3. Export negative tool-failure examples, starting with invalid tool arguments
   and repeated tool-error loops.
4. Hydrate from Langfuse only when needed to fill input/output/transcript text.
5. Emit JSONL with strict schema validation and lineage hashes.
6. Add an LLM annotation pass after deterministic labels are working.

This is actionable in the current workflow because the observability substrate
now exists. The main prerequisite is deciding whether tool result/error text can
be reliably reconstructed from Langfuse or whether we need a small factual
capture extension to `session_history_tool_activity`.
