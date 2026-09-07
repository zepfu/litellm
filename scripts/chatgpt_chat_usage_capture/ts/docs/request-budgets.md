# Request budgets

`BudgetedTransport` wraps one `HistoryTransport` for one collection run. It
serializes account-scoped reads and does not change successful payloads or
HTTP error records. Create a new wrapper for each run; the wrapper is not a
durable scheduler lease and does not ingest or pause an account.

## Defaults

| Control | Default |
| --- | ---: |
| `concurrentReads` | `1` |
| `minGapMs` | `1000` |
| `maxAttemptsPerRun` | `500` |
| `maxRunDurationMs` | `1200000` |
| `requestTimeoutMs` | `30000` |
| `transientRetries` | `2` |
| retry backoff | 250 ms base, 5000 ms cap |
| retry jitter | up to 250 ms, bounded by the cap |

`maxAttemptsPerRun` counts every underlying request, including retries.
`minGapMs` is measured between actual attempt starts, so retries are subject to
the same account read spacing as initial reads. Configuration is validated at
construction; concurrency values other than `1`, non-positive budgets, invalid
durations, and an inverted backoff range are rejected.

## Response and error behavior

- `401`, `403`, and `429` are returned unchanged and are never retried.
  `retry_after` remains available to the adapter and collector.
- HTTP `408` and `5xx` responses receive bounded transient retries. The final
  response is returned unchanged when retries are exhausted.
- Transport failures and request timeouts receive the same bounded retry
  treatment. When no retry remains, the wrapper raises
  `BudgetedTransportPartialError`; it never substitutes an empty response.
- Attempt-budget exhaustion and run-deadline exhaustion raise
  `BudgetedTransportPartialError` and permanently block later reads from that
  wrapper. The collector can persist the partial checkpoint and let the next
  run resume.

The underlying transport must expose `cancel()` or `close()`. On timeout or
explicit cancellation, the wrapper aborts the active request and waits for its
promise to settle before another read can start. `PlaywrightTransport` uses an
abort signal for the active `APIRequestContext` request; its existing browser
`close()` remains the final resource cleanup operation.

## Diagnostics

`getDiagnostics()` returns only run timestamps, attempt/retry totals, the
`identity`/`index`/`head`/`detail` class counts, the last numeric HTTP status,
and the blocking reason. A head read is the offset-zero or offset-omitted
conversation index request. Conversation IDs, query values, response bodies,
headers, cookies, and credentials are not retained in these diagnostics.

The wrapper still calls `assertAllowedRequest` before every read, so it cannot
expand the adapter's GET-only history/session allowlist.
