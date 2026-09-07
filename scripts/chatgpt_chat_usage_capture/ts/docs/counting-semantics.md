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
Draft or inapplicable mappings do not erase an existing classification.
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

## Quota projection

`src/accounting/quota.ts` is a pure projection over reconstructed
`ReconstructedAttempt` values. It does not resolve reset windows, persist
observations, or make provider requests. The caller supplies one resolved
membership value per attempt and policy bucket (`in`, `out`, `ambiguous`, or
`unknown`) plus an overall history coverage state.

The working estimator is
`requested_if_known_else_recorded_final`. A known requested family is selected
first. A missing requested family may use a mapped recorded final family only
when a completed answer exists, and that assessment is labeled
`final_response_inference`. A rejection before generation starts is retained in
the assessment but excluded. The default mode excludes failures after start,
cancellations, unknown acceptance, post-start rejection, conflicting model
families, and unresolved duplicate identities from the working count; it still
reports each as an explicit uncertain-debit category even when a family is
missing, ineligible, or outside the supplied window. The opt-in `include` mode
includes eligible uncertain categories and labels the result accordingly.

Repeated identical records for one attempt identity are deduplicated before
classification. Conflicting records for one identity are retained as one
`conflicting_duplicate_identity` uncertainty assessment and never select an
arbitrary family or contribute to a bucket.
Independent uncertainty categories from owned Chat variants are unioned, so a
conflicting revision does not hide a known failure or cancellation.

Only verified Chat attempts for the expected quota owner contribute. Work,
Codex, other surfaces, shared/imported/copied origins, missing ownership, and
owner mismatches remain visible as exclusions or unclassified activity.
Individual and shared bucket contributions are set-based: one selected attempt
can contribute once to its individual bucket and once to the applicable shared
bucket.

Direct server observations remain separate from local projections. A known
window and complete history coverage are required for a numeric working
remainder and model headroom; otherwise those qualified values are `null`.
When local usage and membership are known but coverage is partial, the
unclamped remainder and any negative discrepancy remain as
`diagnostic_only` evidence, while the qualified remainder stays `null`. The
unclamped remainder preserves negative capacity discrepancies and the
presentation-safe remainder is clamped to zero when qualified. Model headroom
is the minimum of all compatible known individual/shared remainders and is
`null` when any required bucket is unknown.

Definite in-window usage exceeding capacity preserves the negative diagnostic
even when additional attempts have ambiguous membership. That discrepancy is
`diagnostic_only`; unresolved membership still makes qualified remainder and
headroom unknown. Arithmetic fixtures retain weekly history while supplying
distinct daily memberships, so daily rollover does not replenish the weekly
Astra allocation.
