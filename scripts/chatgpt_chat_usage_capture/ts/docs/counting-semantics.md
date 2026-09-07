# Stage-2B counting semantics

Stage 2B is a metadata ledger and raw-model activity report. It is not an
official provider quota meter and does not infer reset windows or billing.

## Observations

An observation is a sanitized source projection. Its identity is
`scope + source_kind + source_id + sanitized_revision_fingerprint`. Replaying
the same sanitized evidence is idempotent. Changed evidence appends a new
immutable observation revision and points to the previous revision.

The collection run ID and observation time are provenance, not evidence
content. They do not cause an identical replay to count again.

## Attempts

An attempt is reconstructed from a user prompt and its reachable generation
chain. Analysis/reasoning nodes, tool calls, tool results, progress frames, and
the terminal answer remain evidence for one generation when their generation
identity and graph linkage support that grouping.

Distinct generation IDs remain distinct attempts, including regenerations.
Request IDs are grouping evidence only and are scoped by conversation and
branch. A collision is recorded as a coverage gap instead of merging two
attempts.

Attempt identity includes:

- local collector account;
- provider;
- verified provider user;
- workspace;
- quota owner;
- conversation;
- validated generation/request/branch identity.

The same upstream generation ID under two accounts therefore produces two
ledger attempts.

## Completion

Only an assistant node with `end_turn=true`, a terminal-success status or an
explicitly empty status, and a non-analysis/non-reasoning/non-tool channel is a
completed answer. In-progress, failed, cancelled, rejected, and analysis-only
nodes remain attempts or uncertain fragments and are never silently counted as
completed answers.

The terminal answer is selected by timestamp, with message ID only as a stable
tie-breaker. Later evidence updates the current attempt projection and appends
an immutable attempt revision.

## Models

Requested, recorded-final, and resolved raw model labels remain separate.
Canonical families are derived through a versioned, reviewed exact mapping.
Unmapped raw labels remain reportable. A mapping change reclassifies retained
attempts and records mapping history; it does not create another attempt.

## Reports

The raw-model report uses a half-open elapsed interval `[start, end)`. It
reports requested raw-model attempts, completed recorded-final raw-model
answers, resolved raw-model observations, mismatches, unknown time, ambiguous
time, excluded surfaces, and shared/imported/copied exclusions independently.
It does not calculate remaining quota, reset periods, or provider charges.
