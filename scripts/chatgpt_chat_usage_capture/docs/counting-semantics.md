# Counting semantics

Independent totals:

- observed attempts by requested family
- completed answers by recorded-final family
- observed model mismatches
- unclassified or ambiguous attempts
- working quota usage estimate by bucket
- server-reported remaining, only when actually stored

Default working estimator: `requested_if_known_else_recorded_final`. Shared buckets use `union_once_per_attempt`: one contribution to the combined bucket per eligible attempt.

Unknown windows use `show_activity_only`. Working used/remaining stay `null`, not `0`. Recording a quota snapshot does not imply a window start.

Generation IDs are stronger attempt identity than request IDs. Distinct generations
that reuse one request ID remain separate attempts; exact repeats of one generation
remain idempotent. Failed, cancelled, and in-progress assistant nodes remain
uncertain until an explicit successful terminal is observed, so they are not
completed answers or default working-estimate contributions.

Mapping changes rebuild derived families from the retained raw model fields. The
current mapping version is applied to the new aggregate revision while prior
attempt projections remain auditable in mapping history; a rebuild does not merely
rename a stale aggregate payload.
