import { describe, expect, it } from "vitest";

import {
  evaluateResetWindow,
  intervalMembership,
  nextRollingExpiry,
  resolveResetWindow,
  windowMembership,
} from "../../src/accounting/windows.js";

const DAY_MS = 24 * 60 * 60 * 1000;

describe("reset-window engine", () => {
  it("selects provider evidence before operator, reviewed, and provisional evidence", () => {
    const resolved = resolveResetWindow({
      asOf: "2026-09-07T12:00:00.000Z",
      evidence: [
        {
          source: "provisional_assumption",
          provenance: "operator estimate",
          rule: {
            type: "rolling_elapsed",
            durationMs: DAY_MS,
          },
        },
        {
          source: "reviewed_rule",
          provenance: "reviewed observations",
          reviewed: true,
          supportedByObservations: true,
          rule: {
            type: "calendar",
            timezone: "America/New_York",
            period: "day",
          },
        },
        {
          source: "operator_explicit",
          provenance: "quota settings page",
          rule: {
            type: "operator_explicit",
            start: "2026-09-07T00:00:00.000Z",
            end: "2026-09-14T00:00:00.000Z",
          },
        },
        {
          source: "provider_explicit",
          provenance: "provider quota observation",
          validated: true,
          observedAt: "2026-09-07T11:00:00.000Z",
          rule: {
            type: "provider_explicit",
            start: "2026-09-07T04:00:00.000Z",
            end: "2026-09-08T04:00:00.000Z",
            windowId: "provider-window-7",
          },
        },
      ],
    });

    expect(resolved.evidenceSource).toBe("provider_explicit");
    expect(resolved.selectedEvidence?.rule.windowId).toBe("provider-window-7");
    expect(resolved.start).toBe("2026-09-07T04:00:00.000Z");
    expect(resolved.end).toBe("2026-09-08T04:00:00.000Z");
    expect(resolved.known).toBe(true);
  });

  it("preserves a null previous boundary when only the provider next reset is known", () => {
    const resolved = resolveResetWindow({
      asOf: "2026-09-07T12:00:00.000Z",
      evidence: [
        {
          source: "provider_explicit",
          provenance: "provider next-reset field",
          validated: true,
          rule: {
            type: "provider_explicit",
            end: "2026-09-08T04:00:00.000Z",
            windowId: "provider-window-8",
          },
        },
      ],
    });

    expect(resolved.start).toBeNull();
    expect(resolved.end).toBe("2026-09-08T04:00:00.000Z");
    expect(resolved.known).toBe(false);
    expect(
      windowMembership({
        window: resolved,
        instant: "2026-09-07T13:00:00.000Z",
      }),
    ).toBe("unknown");
    expect(
      windowMembership({
        window: resolved,
        instant: "2026-09-08T04:00:00.000Z",
      }),
    ).toBe("out");
  });

  it("uses half-open explicit membership and interval ambiguity", () => {
    const window = {
      start: "2026-09-07T00:00:00.000Z",
      end: "2026-09-08T00:00:00.000Z",
    };
    expect(windowMembership({ window, instant: window.start })).toBe("in");
    expect(windowMembership({ window, instant: window.end })).toBe("out");
    expect(
      intervalMembership({
        window,
        time: {
          earliestPossibleAt: "2026-09-07T23:59:00.000Z",
          latestPossibleAt: "2026-09-08T00:01:00.000Z",
        },
      }),
    ).toBe("ambiguous");
    expect(
      intervalMembership({
        window,
        time: {
          earliestPossibleAt: "2026-09-07T01:00:00.000Z",
          latestPossibleAt: "2026-09-07T02:00:00.000Z",
        },
      }),
    ).toBe("in");
  });

  it("steps anchored elapsed windows and opens the next half-open period at a boundary", () => {
    const evaluated = evaluateResetWindow({
      rule: {
        type: "anchored_elapsed",
        anchor: "2026-09-01T06:00:00.000Z",
        durationMs: 7 * DAY_MS,
      },
      asOf: "2026-09-08T06:00:00.000Z",
    });
    expect(evaluated.start).toBe("2026-09-08T06:00:00.000Z");
    expect(evaluated.end).toBe("2026-09-15T06:00:00.000Z");
    expect(evaluated.known).toBe(true);
  });

  it("resolves calendar day boundaries through DST transitions", () => {
    const spring = evaluateResetWindow({
      rule: {
        type: "calendar",
        timezone: "America/New_York",
        period: "day",
      },
      asOf: "2026-03-08T16:00:00.000Z",
    });
    expect(spring.start).toBe("2026-03-08T05:00:00.000Z");
    expect(spring.end).toBe("2026-03-09T04:00:00.000Z");

    const fall = evaluateResetWindow({
      rule: {
        type: "calendar",
        timezone: "America/New_York",
        period: "day",
      },
      asOf: "2026-11-01T17:00:00.000Z",
    });
    expect(fall.start).toBe("2026-11-01T04:00:00.000Z");
    expect(fall.end).toBe("2026-11-02T05:00:00.000Z");
  });

  it("resolves a Monday-start calendar week in the configured timezone", () => {
    const evaluated = evaluateResetWindow({
      rule: {
        type: "calendar",
        timezone: "America/New_York",
        period: "week",
      },
      asOf: "2026-09-09T16:00:00.000Z",
    });
    expect(evaluated.start).toBe("2026-09-07T04:00:00.000Z");
    expect(evaluated.end).toBe("2026-09-14T04:00:00.000Z");
    expect(evaluated.weekStartsOn).toBe(1);
  });

  it("returns event-specific rolling expiry rather than one shared reset", () => {
    const evaluated = evaluateResetWindow({
      rule: {
        type: "rolling_elapsed",
        durationMs: DAY_MS,
      },
      asOf: "2026-09-10T12:00:00.000Z",
      eventTimes: [
        "2026-09-09T13:00:00.000Z",
        "2026-09-10T00:00:00.000Z",
        "2026-09-10T12:00:00.000Z",
      ],
    });
    expect(evaluated.start).toBe("2026-09-09T12:00:00.000Z");
    expect(evaluated.end).toBe("2026-09-10T12:00:00.000Z");
    expect(evaluated.nextExpiryAt).toBe("2026-09-10T13:00:00.000Z");
    expect(
      nextRollingExpiry({
        eventTimes: ["2026-09-08T00:00:00.000Z"],
        durationMs: DAY_MS,
        asOf: "2026-09-10T12:00:00.000Z",
      }),
    ).toBeNull();
  });

  it("keeps unknown policies unknown instead of inventing boundaries", () => {
    const evaluated = resolveResetWindow({
      asOf: "2026-09-07T12:00:00.000Z",
      evidence: [],
    });
    expect(evaluated.type).toBe("unknown");
    expect(evaluated.start).toBeNull();
    expect(evaluated.end).toBeNull();
    expect(evaluated.known).toBe(false);
  });
});
