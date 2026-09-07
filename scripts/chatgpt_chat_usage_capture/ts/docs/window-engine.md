# Reset Window Engine

`src/accounting/windows.ts` is a pure resolver for reset-period boundaries. It
does not read or write the ledger, calculate quota arithmetic, persist policy,
or make provider requests.

## Public interfaces

The primary entry points are:

```ts
resolveResetWindow({
  asOf,
  evidence,
  eventTimes,
}): ResolvedResetWindow

evaluateResetWindow({
  rule,
  asOf,
  eventTimes,
}): EvaluatedResetWindow

windowMembership({ window, instant }): "in" | "out" | "unknown"
intervalMembership({ window, time }): "in" | "out" | "ambiguous" | "unknown"
nextRollingExpiry({ eventTimes, durationMs, asOf }): string | null
```

All input instants may be ISO strings or `Date` values. Returned instants are
UTC ISO strings. `durationMs` is a positive safe integer representing elapsed
milliseconds.

## Evidence precedence

`resolveResetWindow` retains the supplied evidence and selects the strongest
eligible interpretation:

1. `provider_explicit`: `validated: true`, current provider evidence with a
   boundary or verified `windowId`.
2. `operator_explicit`: exact `start` and `end` with non-empty provenance.
3. `reviewed_rule`: a recurring rule marked `reviewed` and
   `supportedByObservations`.
4. `provisional_assumption`: an explicitly labeled provisional recurring rule.
5. `unknown`.

The provider rule may contain only `end` when the provider exposes only a next
reset. The engine preserves `start: null`; it does not infer a previous reset,
a rolling rule, or a timezone from that timestamp.

## Window rules

- `provider_explicit` and `operator_explicit` use explicit UTC boundaries.
- `anchored_elapsed` uses `anchor` (or `start`) plus a fixed elapsed duration.
  The evaluated period containing `asOf` is selected; an exact boundary starts
  the next period.
- `calendar` requires an IANA `timezone` and supports `day` and `week`.
  Weeks start on Monday by default and can use `weekStartsOn` with JavaScript
  weekday numbering. Local midnights are converted through Node's ICU-backed
  `Intl.DateTimeFormat`, so DST days can be 23 or 25 elapsed hours.
- `rolling_elapsed` evaluates `[asOf - duration, asOf)` and derives
  `nextExpiryAt` from individual event timestamps.
- `unknown` has no justified boundaries.

The engine never treats a documented "day" or "week" as an elapsed duration
unless the selected rule says so. It uses half-open `[start, end)` membership:
the start belongs to the window and the end belongs to the following window.
An interval that straddles both boundaries is `ambiguous`, not counted in both.

`nextExpiryAt` is the earliest expiry at or after `asOf` for an event currently
inside a rolling window. Multiple events can therefore have different expiry
times. `asOf` is the evaluation time; history collection freshness remains a
separate concern for the integration layer.

The only timezone facility used here is the Node runtime's installed ICU
implementation behind `Intl.DateTimeFormat`; no timezone library or dependency
was added.
