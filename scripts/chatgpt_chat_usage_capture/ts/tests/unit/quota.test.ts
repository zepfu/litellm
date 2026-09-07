import { describe, expect, it } from "vitest";

import {
  SEED_CHAT_PRO_QUOTA_POLICY,
  estimateQuota,
  type DirectServerQuotaObservation,
  type ResolvedQuotaWindow,
} from "../../src/accounting/quota.js";
import type { ReconstructedAttempt } from "../../src/ledger/types.js";

const ownership = {
  quotaOwnerId: "quota-owner",
  providerUserId: "provider-user",
  workspaceId: "workspace",
};

const scope: ReconstructedAttempt["scope"] = {
  collectorAccountId: "account",
  provider: "openai",
  providerUserId: "provider-user",
  workspaceId: "workspace",
  quotaOwnerId: "quota-owner",
  surface: "chat",
};

describe("pure quota estimator", () => {
  it("should use the requested family once for each matching individual and shared bucket", () => {
    const attempt = makeAttempt({
      attemptId: "astra-request",
      requestedFamily: "astra_pro",
      recordedFinalFamily: "astra_pro",
    });
    const result = estimateQuota({
      attempts: [attempt],
      policy: SEED_CHAT_PRO_QUOTA_POLICY,
      windows: [
        windowFor("astra_weekly", { "astra-request": "in" }),
        windowFor("sol_daily", { "astra-request": "out" }),
        windowFor("pro_combined_daily", { "astra-request": "in" }),
      ],
      coverage: "complete",
      ownership,
    });

    expect(result.buckets.astra_weekly?.workingUsageEstimate).toBe(1);
    expect(result.buckets.sol_daily?.workingUsageEstimate).toBe(0);
    expect(result.buckets.pro_combined_daily?.workingUsageEstimate).toBe(1);
    expect(result.buckets.pro_combined_daily?.countedAttemptIds).toEqual([
      "astra-request",
    ]);
    expect(result.assessments[0]).toMatchObject({
      selectedFamily: "astra_pro",
      basis: "requested",
      contributedBucketIds: ["astra_weekly", "pro_combined_daily"],
    });
  });

  it("should label a completed final-family fallback without filling raw requested evidence", () => {
    const attempt = makeAttempt({
      attemptId: "final-only",
      requestedFamily: null,
      recordedFinalFamily: "sol_pro",
      requestedModelRaw: null,
    });
    const result = estimateQuota({
      attempts: [attempt],
      windows: [
        windowFor("astra_weekly", { "final-only": "out" }),
        windowFor("sol_daily", { "final-only": "in" }),
        windowFor("pro_combined_daily", { "final-only": "in" }),
      ],
      coverage: "complete",
      ownership,
    });

    expect(attempt.requestedFamily).toBeNull();
    expect(result.assessments[0]).toMatchObject({
      selectedFamily: "sol_pro",
      basis: "final_response_inference",
    });
    expect(result.buckets.sol_daily?.workingUsageEstimate).toBe(1);
    expect(result.buckets.pro_combined_daily?.workingUsageEstimate).toBe(1);
  });

  it("should exclude rejected-before-start and retain other uncertain debit categories", () => {
    const attempts = [
      makeAttempt({
        attemptId: "rejected",
        requestedFamily: "astra_pro",
        generationStarted: false,
        completedAnswer: false,
        outcome: "rejected_before_start",
      }),
      makeAttempt({
        attemptId: "failed",
        requestedFamily: "astra_pro",
        outcome: "failed_after_start",
      }),
      makeAttempt({
        attemptId: "cancelled",
        requestedFamily: "astra_pro",
        outcome: "cancelled_after_start",
      }),
      makeAttempt({
        attemptId: "unknown",
        requestedFamily: "astra_pro",
        outcome: "completion_unknown",
      }),
      makeAttempt({
        attemptId: "rejected-after-start",
        requestedFamily: "astra_pro",
        outcome: "rejected_after_start",
      }),
      makeAttempt({
        attemptId: "conflicting",
        requestedFamily: "astra_pro",
        recordedFinalFamily: "sol_pro",
      }),
      makeAttempt({
        attemptId: "unresolved",
        requestedFamily: "astra_pro",
        identityBasis: "unresolved",
      }),
    ];
    const windows = [
      windowFor("astra_weekly", Object.fromEntries(
        attempts.map((attempt) => [attempt.attemptId, "in"]),
      )),
      windowFor("sol_daily", {}),
      windowFor("pro_combined_daily", Object.fromEntries(
        attempts.map((attempt) => [attempt.attemptId, "in"]),
      )),
    ];

    const excluded = estimateQuota({
      attempts,
      windows,
      coverage: "complete",
      ownership,
    });
    expect(excluded.rejectedBeforeStartAttemptIds).toEqual(["rejected"]);
    expect(excluded.unknownDebitAttempts).toBe(6);
    expect(excluded.uncertainDebitCategories.failed_after_start).toEqual([
      "failed",
    ]);
    expect(excluded.uncertainDebitCategories.cancelled_after_start).toEqual([
      "cancelled",
    ]);
    expect(excluded.uncertainDebitCategories.unknown_acceptance).toEqual([
      "unknown",
    ]);
    expect(excluded.uncertainDebitCategories.rejected_after_start).toEqual([
      "rejected-after-start",
    ]);
    expect(excluded.uncertainDebitCategories.conflicting_models).toEqual([
      "conflicting",
    ]);
    expect(
      excluded.uncertainDebitCategories.unresolved_duplicate_identity,
    ).toEqual(["unresolved"]);
    expect(excluded.buckets.astra_weekly?.workingUsageEstimate).toBe(0);

    const included = estimateQuota({
      attempts,
      windows,
      coverage: "complete",
      ownership,
      uncertainAttemptMode: "include",
    });
    expect(included.buckets.astra_weekly?.workingUsageEstimate).toBe(6);
    expect(included.buckets.pro_combined_daily?.workingUsageEstimate).toBe(6);
    expect(included.uncertainAttemptMode).toBe("include");
  });

  it("should retain uncertainty before family, eligibility, window, and start gates", () => {
    const attempts = [
      makeAttempt({
        attemptId: "failed-no-family",
        requestedFamily: null,
        recordedFinalFamily: null,
        requestedModelRaw: null,
        recordedFinalModelRaw: null,
        generationStarted: true,
        completedAnswer: false,
        outcome: "failed_after_start",
      }),
      makeAttempt({
        attemptId: "failed-ineligible",
        requestedFamily: "other_family",
        recordedFinalFamily: "other_family",
        generationStarted: true,
        completedAnswer: false,
        outcome: "failed_after_start",
      }),
      makeAttempt({
        attemptId: "failed-out-of-window",
        outcome: "failed_after_start",
      }),
      makeAttempt({
        attemptId: "unknown-no-start",
        requestedFamily: null,
        recordedFinalFamily: null,
        requestedModelRaw: null,
        recordedFinalModelRaw: null,
        generationStarted: false,
        completedAnswer: false,
        outcome: "completion_unknown",
      }),
      makeAttempt({
        attemptId: "unresolved-no-start",
        requestedFamily: null,
        recordedFinalFamily: null,
        requestedModelRaw: null,
        recordedFinalModelRaw: null,
        generationStarted: false,
        completedAnswer: false,
        identityBasis: "unresolved",
        outcome: "unresolved",
      }),
    ];
    const result = estimateQuota({
      attempts,
      windows: [
        windowFor("astra_weekly", { "failed-out-of-window": "out" }),
        windowFor("pro_combined_daily", { "failed-out-of-window": "out" }),
      ],
      coverage: "complete",
      ownership,
    });

    expect(result.unknownDebitAttempts).toBe(5);
    expect(result.unknownDebitAttemptIds).toEqual([
      "failed-ineligible",
      "failed-no-family",
      "failed-out-of-window",
      "unknown-no-start",
      "unresolved-no-start",
    ]);
    expect(result.uncertainDebitCategories.failed_after_start).toEqual([
      "failed-ineligible",
      "failed-no-family",
      "failed-out-of-window",
    ]);
    expect(result.uncertainDebitCategories.unknown_acceptance).toEqual([
      "unknown-no-start",
      "unresolved-no-start",
    ]);
    expect(
      result.uncertainDebitCategories.unresolved_duplicate_identity,
    ).toEqual(["unresolved-no-start"]);
    expect(result.assessments).toMatchObject([
      {
        attemptId: "failed-ineligible",
        disposition: "unclassified",
        uncertainDebitCategories: ["failed_after_start"],
        unclassifiedReasons: ["ineligible_family"],
      },
      {
        attemptId: "failed-no-family",
        disposition: "unclassified",
        uncertainDebitCategories: ["failed_after_start"],
        unclassifiedReasons: ["unknown_model"],
      },
      {
        attemptId: "failed-out-of-window",
        disposition: "out_of_window",
        uncertainDebitCategories: ["failed_after_start"],
      },
      {
        attemptId: "unknown-no-start",
        disposition: "unclassified",
        uncertainDebitCategories: ["unknown_acceptance"],
        unclassifiedReasons: ["not_generation_started"],
      },
      {
        attemptId: "unresolved-no-start",
        disposition: "unclassified",
        uncertainDebitCategories: [
          "unknown_acceptance",
          "unresolved_duplicate_identity",
        ],
        unclassifiedReasons: ["not_generation_started"],
      },
    ]);
  });

  it("should deduplicate identical attempt identities and quarantine conflicting duplicates", () => {
    const identical = makeAttempt({
      attemptId: "identical",
      requestedFamily: "astra_pro",
      recordedFinalFamily: "astra_pro",
    });
    const conflictingA = makeAttempt({
      attemptId: "conflicting-duplicate",
      requestedFamily: "astra_pro",
      recordedFinalFamily: "astra_pro",
    });
    const conflictingB = makeAttempt({
      attemptId: "conflicting-duplicate",
      requestedFamily: "sol_pro",
      recordedFinalFamily: "sol_pro",
    });
    const result = estimateQuota({
      attempts: [conflictingB, identical, conflictingA, identical],
      windows: [
        windowFor("astra_weekly", {
          identical: "in",
          "conflicting-duplicate": "in",
        }),
        windowFor("pro_combined_daily", {
          identical: "in",
          "conflicting-duplicate": "in",
        }),
      ],
      coverage: "complete",
      ownership,
    });

    expect(result.assessments.map((assessment) => assessment.attemptId)).toEqual([
      "conflicting-duplicate",
      "identical",
    ]);
    expect(result.countedAttemptIds).toEqual(["identical"]);
    expect(result.buckets.astra_weekly?.workingUsageEstimate).toBe(1);
    expect(result.buckets.pro_combined_daily?.workingUsageEstimate).toBe(1);
    expect(result.unknownDebitAttempts).toBe(1);
    expect(result.unknownDebitAttemptIds).toEqual(["conflicting-duplicate"]);
    expect(result.unclassifiedAttempts).toBe(1);
    expect(result.unclassifiedAttemptIds).toEqual(["conflicting-duplicate"]);
    expect(
      result.uncertainDebitCategories.conflicting_duplicate_identity,
    ).toEqual(["conflicting-duplicate"]);
    expect(
      result.unclassifiedReasons.conflicting_duplicate_identity,
    ).toEqual(["conflicting-duplicate"]);
    expect(result.assessments[0]).toMatchObject({
      attemptId: "conflicting-duplicate",
      disposition: "unclassified",
      selectedFamily: null,
      applicableBucketIds: [],
      contributedBucketIds: [],
      membershipByBucket: {},
      uncertainDebitCategories: ["conflicting_duplicate_identity"],
      unclassifiedReasons: ["conflicting_duplicate_identity"],
    });
  });

  it("should exclude non-Chat, unattributed, unknown-owner, and other-owner attempts", () => {
    const attempts = [
      makeAttempt({
        attemptId: "work",
        surface: "work",
        scope: { ...scope, surface: "work" },
      }),
      makeAttempt({
        attemptId: "unknown-owner",
        scope: { ...scope, quotaOwnerId: null },
      }),
      makeAttempt({
        attemptId: "other-owner",
        scope: { ...scope, quotaOwnerId: "other-owner" },
      }),
      makeAttempt({
        attemptId: "shared",
        origin: "shared",
      }),
    ];
    const result = estimateQuota({
      attempts,
      windows: [
        windowFor("astra_weekly", Object.fromEntries(
          attempts.map((attempt) => [attempt.attemptId, "in"]),
        )),
        windowFor("pro_combined_daily", Object.fromEntries(
          attempts.map((attempt) => [attempt.attemptId, "in"]),
        )),
      ],
      coverage: "complete",
      ownership,
    });

    expect(result.countedAttemptIds).toEqual([]);
    expect(result.unclassifiedAttempts).toBe(4);
    expect(result.unclassifiedReasons.unknown_surface).toEqual(["work"]);
    expect(result.unclassifiedReasons.unknown_ownership).toEqual([
      "unknown-owner",
    ]);
    expect(result.unclassifiedReasons.ownership_mismatch).toEqual([
      "other-owner",
    ]);
    expect(result.unclassifiedReasons.unattributed_origin).toEqual(["shared"]);
  });

  it("should keep direct server observations separate and expose negative unclamped discrepancies", () => {
    const policy = {
      policyId: "fixture-policy",
      version: "fixture-v1",
      buckets: [
        {
          bucketId: "individual",
          kind: "individual" as const,
          eligibleFamilies: ["fixture_family"],
          capacity: 1,
        },
        {
          bucketId: "shared",
          kind: "shared" as const,
          eligibleFamilies: ["fixture_family"],
          capacity: 2,
        },
      ],
    };
    const attempts = ["one", "two", "three"].map((attemptId) =>
      makeAttempt({
        attemptId,
        requestedFamily: "fixture_family",
        recordedFinalFamily: "fixture_family",
      }),
    );
    const serverObservation: DirectServerQuotaObservation = {
      bucketId: "individual",
      windowId: "individual-window",
      remaining: 99,
      observedAt: "2026-09-07T12:00:00.000Z",
    };
    const result = estimateQuota({
      policy,
      attempts,
      windows: [
        windowFor("individual", Object.fromEntries(
          attempts.map((attempt) => [attempt.attemptId, "in"]),
        ), "individual-window"),
        windowFor("shared", Object.fromEntries(
          attempts.map((attempt) => [attempt.attemptId, "in"]),
        ), "shared-window"),
      ],
      coverage: "complete",
      ownership,
      serverObservations: [serverObservation],
    });

    expect(result.buckets.individual).toMatchObject({
      workingUsageEstimate: 3,
      workingRemainingEstimate: 0,
      workingRemainingUnclamped: -2,
      discrepancy: {
        kind: "local_usage_exceeds_capacity",
        excess: 2,
        qualification: "qualified",
      },
      serverReportedRemaining: serverObservation,
    });
    expect(result.buckets.shared?.workingRemainingUnclamped).toBe(-1);
    expect(result.workingHeadroomByFamily.fixture_family).toBe(0);
  });

  it("should make remainders unknown for incomplete coverage or unresolved membership", () => {
    const attempt = makeAttempt({
      attemptId: "known",
      requestedFamily: "astra_pro",
    });
    const windows = [
      windowFor("astra_weekly", { known: "in" }),
      windowFor("pro_combined_daily", {}),
    ];
    const partial = estimateQuota({
      attempts: [attempt],
      windows,
      coverage: "partial",
      ownership,
    });
    expect(partial.buckets.astra_weekly).toMatchObject({
      workingUsageEstimate: 1,
      workingRemainingEstimate: null,
      workingRemainingUnclamped: 199,
    });
    expect(partial.workingHeadroomByFamily.astra_pro).toBeNull();

    const diagnosticPolicy = {
      policyId: "diagnostic-policy",
      version: "diagnostic-v1",
      buckets: [
        {
          bucketId: "diagnostic",
          kind: "individual" as const,
          eligibleFamilies: ["fixture_family"],
          capacity: 1,
        },
      ],
    };
    const diagnosticAttempts = ["one", "two", "three"].map((attemptId) =>
      makeAttempt({
        attemptId,
        requestedFamily: "fixture_family",
        recordedFinalFamily: "fixture_family",
      }),
    );
    const diagnostic = estimateQuota({
      policy: diagnosticPolicy,
      attempts: diagnosticAttempts,
      windows: [
        windowFor(
          "diagnostic",
          Object.fromEntries(
            diagnosticAttempts.map((attempt) => [attempt.attemptId, "in"]),
          ),
        ),
      ],
      coverage: "partial",
      ownership,
    });
    expect(diagnostic.buckets.diagnostic).toMatchObject({
      workingUsageEstimate: 3,
      workingRemainingEstimate: null,
      workingRemainingUnclamped: -2,
      discrepancy: {
        kind: "local_usage_exceeds_capacity",
        excess: 2,
        qualification: "diagnostic_only",
      },
    });
    expect(diagnostic.workingHeadroomByFamily.fixture_family).toBeNull();

    const completeWithUnknownMembership = estimateQuota({
      attempts: [attempt],
      windows,
      coverage: "complete",
      ownership,
    });
    expect(
      completeWithUnknownMembership.buckets.pro_combined_daily,
    ).toMatchObject({
      workingUsageEstimate: 0,
      workingRemainingEstimate: null,
      workingRemainingUnclamped: null,
    });
    expect(
      completeWithUnknownMembership.windowMembershipUnknownAttemptIds,
    ).toEqual(["known"]);
  });

  it("should satisfy the section 21 seven-day arithmetic fixtures", () => {
    const firstSchedule = [
      [170, 30],
      [170, 30],
      [170, 30],
      [170, 30],
      [170, 30],
      [170, 30],
      [170, 20],
    ] as const;
    const firstTotals = { sol: 0, astra: 0, combined: 0 };

    for (const [dayIndex, [solCount, astraCount]] of firstSchedule.entries()) {
      const day = arithmeticDay("first", dayIndex + 1, solCount, astraCount);
      const result = estimateArithmeticDay(day);
      const sol = result.buckets.sol_daily?.workingUsageEstimate ?? null;
      const astra = result.buckets.astra_weekly?.workingUsageEstimate ?? null;
      const combined =
        result.buckets.pro_combined_daily?.workingUsageEstimate ?? null;

      expect(sol).toBe(solCount);
      expect(astra).toBe(astraCount);
      expect(combined).toBe(solCount + astraCount);
      expect(sol ?? Infinity).toBeLessThanOrEqual(170);
      expect(combined ?? Infinity).toBeLessThanOrEqual(200);
      firstTotals.sol += sol ?? 0;
      firstTotals.astra += astra ?? 0;
      firstTotals.combined += combined ?? 0;
    }

    expect(firstTotals).toEqual({ sol: 1190, astra: 200, combined: 1390 });

    let secondCombinedTotal = 0;
    for (let day = 1; day <= 7; day += 1) {
      const dayFixture =
        day === 1
          ? arithmeticDay("second", day, 0, 200)
          : arithmeticDay("second", day, 170, 0);
      const result = estimateArithmeticDay(dayFixture);
      const sol = result.buckets.sol_daily?.workingUsageEstimate ?? null;
      const combined =
        result.buckets.pro_combined_daily?.workingUsageEstimate ?? null;

      if (day === 1) {
        expect(sol).toBe(0);
        expect(combined).toBe(200);
      } else {
        expect(sol).toBe(170);
        expect(combined).toBe(170);
      }
      secondCombinedTotal += combined ?? 0;
    }

    expect(secondCombinedTotal).toBe(1220);
  });
});

function windowFor(
  bucketId: string,
  membershipByAttemptId: Readonly<
    Record<string, "in" | "out" | "ambiguous" | "unknown">
  >,
  windowId = `${bucketId}-window`,
): ResolvedQuotaWindow {
  return {
    bucketId,
    windowId,
    status: "known",
    membershipByAttemptId,
  };
}

function makeAttempt(
  overrides: Partial<ReconstructedAttempt> = {},
): ReconstructedAttempt {
  return {
    attemptId: "attempt",
    conversationId: "conversation",
    identityBasis: "generation",
    timeBasis: "user_message",
    attemptTime: "2026-09-07T12:00:00.000Z",
    earliestPossibleAt: "2026-09-07T12:00:00.000Z",
    latestPossibleAt: "2026-09-07T12:00:00.000Z",
    requestedModelRaw: "requested-model",
    requestedModeRaw: null,
    requestedReasoningEffortRaw: null,
    recordedFinalModelRaw: "final-model",
    resolvedModelRaw: null,
    requestedFamily: "astra_pro",
    recordedFinalFamily: "astra_pro",
    resolvedFamily: null,
    mappingVersion: "mapping-v1",
    outcome: "completed",
    completedAnswer: true,
    generationStarted: true,
    surface: "chat",
    origin: null,
    aliases: [],
    evidenceMessageIds: [],
    revision: 1,
    warnings: [],
    scope,
    ...overrides,
  };
}

function arithmeticDay(
  fixture: string,
  day: number,
  solCount: number,
  astraCount: number,
): {
  attempts: ReconstructedAttempt[];
  solIds: string[];
  astraIds: string[];
} {
  const solIds = Array.from({ length: solCount }, (_, index) =>
    `${fixture}-day-${day}-sol-${index}`,
  );
  const astraIds = Array.from({ length: astraCount }, (_, index) =>
    `${fixture}-day-${day}-astra-${index}`,
  );
  const attempts = [
    ...solIds.map((attemptId) =>
      makeAttempt({
        attemptId,
        requestedFamily: "sol_pro",
        recordedFinalFamily: "sol_pro",
      }),
    ),
    ...astraIds.map((attemptId) =>
      makeAttempt({
        attemptId,
        requestedFamily: "astra_pro",
        recordedFinalFamily: "astra_pro",
      }),
    ),
  ];
  return { attempts, solIds, astraIds };
}

function estimateArithmeticDay(day: {
  attempts: ReconstructedAttempt[];
  solIds: string[];
  astraIds: string[];
}) {
  const combinedIds = [...day.solIds, ...day.astraIds];
  return estimateQuota({
    attempts: day.attempts,
    windows: [
      windowFor("astra_weekly", membershipFor(day.astraIds)),
      windowFor("sol_daily", membershipFor(day.solIds)),
      windowFor("pro_combined_daily", membershipFor(combinedIds)),
    ],
    coverage: "complete",
    ownership,
  });
}

function membershipFor(
  attemptIds: ReadonlyArray<string>,
): Record<string, "in"> {
  return Object.fromEntries(attemptIds.map((attemptId) => [attemptId, "in"]));
}
