# Reset-window semantics

Supported types: `provider_explicit`, `operator_explicit`, `anchored_elapsed`, `calendar`, `rolling_elapsed`, `unknown`.

All windows are half-open `[start, end)`. Calendar recurrence uses the configured timezone (`America/New_York` by default). A documented “day” or “week” is a period hint, not an automatic midnight or Monday anchor.

`anchored_elapsed` steps from a reviewed UTC start by a fixed duration. `rolling_elapsed` is a sliding duration ending at the report as-of instant. Crossing a boundary opens a new window; the ledger is never zeroed.
