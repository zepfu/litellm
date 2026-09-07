import { describe, expect, it } from "vitest";

import {
  defaultBackfillRange,
  makeRange,
  parseDuration,
  resolveRequestedRange,
} from "../../src/history/range.js";

describe("Stage-2A history ranges", () => {
  it("defaults backfill to exactly fourteen elapsed days", () => {
    const now = new Date("2026-09-07T12:00:00.000Z");
    expect(defaultBackfillRange(now)).toEqual({
      start: "2026-08-24T12:00:00.000Z",
      end: "2026-09-07T12:00:00.000Z",
    });
  });

  it("accepts arbitrary elapsed durations", () => {
    expect(parseDuration("48h")).toBe(48 * 60 * 60 * 1000);
    expect(parseDuration("90m")).toBe(90 * 60 * 1000);
    expect(
      resolveRequestedRange({
        mode: "backfill",
        now: new Date("2026-09-07T12:00:00.000Z"),
        range: makeRange(
          new Date("2026-09-01T00:00:00.000Z"),
          new Date("2026-09-02T00:00:00.000Z"),
        ),
      }),
    ).toEqual({
      start: "2026-09-01T00:00:00.000Z",
      end: "2026-09-02T00:00:00.000Z",
    });
  });

  it("requires reconciliation to carry an explicit range", () => {
    expect(() =>
      resolveRequestedRange({
        mode: "reconciliation",
        now: new Date("2026-09-07T12:00:00.000Z"),
      }),
    ).toThrow("reconciliation requires an explicit range");
  });
});
