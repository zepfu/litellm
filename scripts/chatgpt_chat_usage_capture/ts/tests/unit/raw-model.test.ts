import { describe, expect, it } from "vitest";

import { buildRawModelReport } from "../../src/accounting/raw-model.js";
import type { Ledger } from "../../src/ledger/store.js";
import type { LedgerScope } from "../../src/ledger/types.js";

const scope: LedgerScope = {
  collectorAccountId: "account-raw-model",
  provider: "openai",
  providerUserId: "provider-user",
  workspaceId: "workspace",
  quotaOwnerId: "quota-owner",
  surface: "chat",
};

const reportStart = "2026-09-07T11:00:00.000Z";
const reportEnd = "2026-09-07T12:00:00.000Z";

describe("raw-model accounting", () => {
  it("should include a point exactly at report start and exclude the end", () => {
    const report = buildRawModelReport(
      fakeLedger([
        attempt({
          attemptId: "at-start",
          attemptTime: reportStart,
          earliestPossibleAt: reportStart,
          latestPossibleAt: reportStart,
          requestedModelRaw: "slug-start",
        }),
        attempt({
          attemptId: "bounded-start",
          earliestPossibleAt: reportStart,
          latestPossibleAt: reportStart,
          requestedModelRaw: "slug-bounded",
        }),
        attempt({
          attemptId: "at-end",
          attemptTime: reportEnd,
          earliestPossibleAt: reportEnd,
          latestPossibleAt: reportEnd,
          requestedModelRaw: "slug-end",
        }),
      ]),
      scope,
      { durationMs: 60 * 60 * 1000, now: reportEnd },
    );

    expect(report.observedAttemptsByRequestedModel).toEqual({
      "slug-bounded": 1,
      "slug-start": 1,
    });
    expect(report.attemptIds).toEqual(["at-start", "bounded-start"]);
    expect(report.possibleAttemptIds).toEqual([]);
  });

  it("should keep straddling, unknown-time, and unresolved attempts out of definite totals", () => {
    const attempts = [
      attempt({
        attemptId: "definite",
        attemptTime: reportStart,
        earliestPossibleAt: reportStart,
        latestPossibleAt: reportStart,
        requestedModelRaw: "slug-definite",
      }),
      attempt({
        attemptId: "straddling",
        attemptTime: null,
        earliestPossibleAt: "2026-09-07T10:59:00.000Z",
        latestPossibleAt: "2026-09-07T11:01:00.000Z",
        requestedModelRaw: "slug-straddling",
      }),
      attempt({
        attemptId: "unknown-time",
        requestedModelRaw: "slug-unknown-time",
        attemptTime: null,
        earliestPossibleAt: null,
        latestPossibleAt: null,
      }),
      attempt({
        attemptId: "unresolved",
        attemptTime: reportStart,
        earliestPossibleAt: reportStart,
        latestPossibleAt: reportStart,
        identityBasis: "unresolved",
        outcome: "unresolved",
        requestedModelRaw: "slug-unresolved",
      }),
    ];

    const report = buildRawModelReport(
      fakeLedger(attempts),
      scope,
      { durationMs: 60 * 60 * 1000, now: reportEnd },
    );
    expect(report.observedAttemptsByRequestedModel).toEqual({
      "slug-definite": 1,
    });
    expect(report.possibleAttemptIds).toEqual([
      "straddling",
      "unknown-time",
      "unresolved",
    ]);
    expect(report.possibleAttemptsByRequestedModel).toEqual({
      "slug-straddling": 1,
      "slug-unknown-time": 1,
      "slug-unresolved": 1,
    });
    expect(report.ambiguousTimeAttempts).toBe(1);
    expect(report.unknownTimeAttempts).toBe(1);
    expect(report.unresolvedAttempts).toBe(1);
    expect(report.unclassifiedAttempts).toBe(3);

    const previousReport = buildRawModelReport(
      fakeLedger(attempts),
      scope,
      { durationMs: 60 * 60 * 1000, now: reportStart },
    );
    expect(previousReport.observedAttemptsByRequestedModel).toEqual({});
    expect(previousReport.possibleAttemptIds).toContain("straddling");
  });

  it("should compare mapped families while retaining raw slug differences", () => {
    const report = buildRawModelReport(
      fakeLedger([
        attempt({
          attemptId: "same-family",
          attemptTime: reportStart,
          requestedModelRaw: "slug-requested",
          recordedFinalModelRaw: "slug-final",
          requestedFamily: "astra_pro",
          recordedFinalFamily: "astra_pro",
        }),
        attempt({
          attemptId: "different-family",
          attemptTime: reportStart,
          requestedModelRaw: "slug-requested-2",
          recordedFinalModelRaw: "slug-final-2",
          requestedFamily: "astra_pro",
          recordedFinalFamily: "sol_pro",
        }),
      ]),
      scope,
      { durationMs: 60 * 60 * 1000, now: reportEnd },
    );

    expect(report.modelMismatches).toBe(1);
    expect(report.rawSlugDifferences).toBe(2);
  });

  it("should count one unclassified attempt across overlapping uncertainty categories", () => {
    const report = buildRawModelReport(
      fakeLedger([
        attempt({
          attemptId: "overlapping-uncertainty",
          surface: "work",
          attemptTime: null,
          earliestPossibleAt: null,
          latestPossibleAt: null,
          requestedModelRaw: null,
          recordedFinalModelRaw: null,
          resolvedModelRaw: null,
        }),
      ]),
      scope,
      { durationMs: 60 * 60 * 1000, now: reportEnd },
    );

    expect(report.unclassifiedAttempts).toBe(1);
    expect(report.unknownTimeAttempts).toBe(1);
    expect(report.unknownModelAttempts).toBe(1);
    expect(report.excludedSurfaceAttempts).toBe(1);
  });
});

function fakeLedger(
  attempts: Array<Record<string, unknown>>,
): Ledger {
  return {
    listAttempts: () => attempts,
    coverageGaps: () => [],
  } as unknown as Ledger;
}

function attempt(
  overrides: Partial<Record<string, unknown>> = {},
): Record<string, unknown> {
  return {
    attemptId: "attempt",
    attemptTime: reportStart,
    earliestPossibleAt: reportStart,
    latestPossibleAt: reportStart,
    requestedModelRaw: "slug-requested",
    recordedFinalModelRaw: "slug-final",
    resolvedModelRaw: "slug-resolved",
    requestedFamily: "astra_pro",
    recordedFinalFamily: "astra_pro",
    resolvedFamily: "astra_pro",
    identityBasis: "generation",
    outcome: "completed",
    completedAnswer: true,
    surface: "chat",
    origin: null,
    ...overrides,
  };
}
