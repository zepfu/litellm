# Stage-2B counting semantics

Stage 2B is a metadata ledger and raw-model activity report. It is not an
official provider quota meter and does not infer reset windows or billing.

## Observations

An observation is a sanitized source projection. Its identity includes activity
scope, source kind, source ID, occurrence revision, and sanitized fingerprint.
Consecutive identical evidence from the same collector deduplicates. Changed
evidence, including an A-B-A reversion or a different collector's observation,
appends an immutable revision pointing to the previous occurrence.

The collection run ID and observation time are provenance, not evidence
content. They do not cause an identical replay to count again.

Raw observation freshness is tracked separately from the current projection.
`activity_provenance.last_seen_at` is the ordering watermark for message and
attempt evidence, so an identical replay advances freshness without creating a
revision. The `updated_at` value on a current message or attempt row records
the projection update instead; mapping reclassification may move that
timestamp without making later raw evidence stale.

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

- provider;
- verified provider user;
- workspace;
- quota owner;
- surface;
- conversation;
- validated generation/request/branch identity.

The same upstream activity collected by two local accounts for one verified
owner converges on one active attempt with separate collector provenance.
Different owners remain isolated. Unverified identity is collector-local.
Later strong linkage can retire a provisional duplicate while retaining its
revision history and transferring evidence and aliases to the active attempt.
Regenerations do not inherit prompt model/time without linkage, and a final-only
response does not invent request-time bounds.

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
Unmapped raw labels remain reportable. Published mappings are immutable;
prospective versions and bounded historical corrections apply at event time.
Changed raw evidence under an inapplicable mapping is re-evaluated with the
applicable stored mapping; when none applies, all family fields remain
unresolved with an explicit warning rather than retaining a stale family.
Collector-specific overrides are validated at canonical-owner scope, so
conflicting overrides for collectors sharing one owner are rejected.
Reclassification preserves warnings and mapping provenance and does not create
another attempt or reactivate a retired duplicate.

## Reports

The raw-model report uses a half-open elapsed interval `[start, end)`. A point
exactly at `start` is included and a point exactly at `end` is excluded.
Definite raw-model totals include only eligible Chat attempts whose time
membership is definite and whose fragment is not unresolved. Straddling
intervals, unknown-time attempts, and unresolved fragments remain in
`possibleAttemptIds` and the possible raw-model maps when their evidence is
available; possible membership is qualified evidence and must not be added to
definite totals or treated as a bucket assignment.

Requested and recorded-final family mismatches compare the mapped family
fields. `rawSlugDifferences` separately preserves cases where the raw
requested and recorded-final slugs differ even when both map to one family.
Unclassified attempts are counted by distinct attempt ID. The uncertainty
category counters can overlap for one attempt and therefore are not additive.

The report also exposes unknown time, ambiguous time, unresolved fragments,
unknown model evidence, excluded surfaces, and shared/imported/copied
exclusions independently. It does not calculate remaining quota, reset
periods, or provider charges.
