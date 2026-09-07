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
