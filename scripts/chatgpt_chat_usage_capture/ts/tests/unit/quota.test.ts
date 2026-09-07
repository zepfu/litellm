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
      workingRemainingUnclamped: null,
    });
    expect(partial.workingHeadroomByFamily.astra_pro).toBeNull();

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
