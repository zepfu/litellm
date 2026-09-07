import { describe, expect, it } from "vitest";

import type { QuotaEstimate } from "../../src/accounting/quota.js";
import type { RawModelReport } from "../../src/accounting/raw-model.js";
import {
  buildReportExport,
  renderCsv,
  renderMarkdown,
  renderReport,
  type ReportExportInput,
} from "../../src/reporting/export.js";

const report: RawModelReport = {
  accountId: "account-report",
  scopeKey: "scope-report",
  surface: "chat",
  durationMs: 60 * 60 * 1000,
  start: "2026-09-07T11:00:00.000Z",
  end: "2026-09-07T12:00:00.000Z",
  observedAttemptsByRequestedModel: {
    "=SUM(A1:A2), \"quoted\"": 1,
    "model|pipe": 2,
  },
  completedAnswersByRecordedFinalModel: {
    "model-final": 1,
  },
  observedAttemptsByResolvedModel: {
    "model-resolved": 1,
  },
  modelMismatches: 1,
  rawSlugDifferences: 2,
  includedAttempts: 2,
  possibleAttemptIds: ["possible-1"],
  possibleAttemptsByRequestedModel: {
    "model-possible": 1,
  },
  possibleCompletedAnswersByRecordedFinalModel: {},
  possibleAttemptsByResolvedModel: {},
  ambiguousTimeAttempts: 1,
  unknownTimeAttempts: 1,
  unresolvedAttempts: 0,
  unknownModelAttempts: 0,
  excludedSurfaceAttempts: 0,
  excludedOriginAttempts: 0,
  unclassifiedAttempts: 1,
  attemptIds: ["attempt-1", "attempt-2"],
  coverageGaps: [
    {
      gap_id: "gap-1",
      source_kind: "history",
      source_id: "page|1",
      reason: "partial|page",
      state: "open",
      first_seen_at: "2026-09-07T11:01:00.000Z",
      last_seen_at: "2026-09-07T11:02:00.000Z",
      details: {
        coverage: "partial",
        warnings: ["missing|continuation"],
        content: "must-not-export",
      },
      content: "must-not-export",
    },
  ],
  label: "Observed raw-model activity; not an official provider quota balance",
};

const quotaEstimate: QuotaEstimate = {
  policyId: "policy-1",
  policyVersion: "policy-v1",
  workingEstimator: "requested_if_known_else_recorded_final",
  uncertainAttemptMode: "exclude",
  coverage: "partial",
  buckets: {
    "bucket-1": {
      bucketId: "bucket-1",
      kind: "individual",
      eligibleFamilies: ["astra_pro"],
      capacity: 10,
      windowId: "window-1",
      workingUsageEstimate: 12,
      workingRemainingEstimate: null,
      workingRemainingUnclamped: -2,
      discrepancy: {
        kind: "local_usage_exceeds_capacity",
        excess: 2,
        qualification: "diagnostic_only",
      },
      serverReportedRemaining: {
        bucketId: "bucket-1",
        windowId: "window-1",
        remaining: 8,
        observedAt: "2026-09-07T11:59:00.000Z",
      },
      countedAttemptIds: ["attempt-1"],
      unknownDebitAttemptIds: ["attempt-2"],
      unknownMembershipAttemptIds: [],
    },
  },
  workingUsageEstimateByBucket: {
    "bucket-1": 12,
  },
  workingRemainingEstimateByBucket: {
    "bucket-1": null,
  },
  workingRemainingUnclampedByBucket: {
    "bucket-1": -2,
  },
  serverReportedRemainingByBucket: {
    "bucket-1": {
      bucketId: "bucket-1",
      windowId: "window-1",
      remaining: 8,
      observedAt: "2026-09-07T11:59:00.000Z",
    },
  },
  workingHeadroomByFamily: {
    astra_pro: null,
  },
  countedAttemptIds: ["attempt-1"],
  rejectedBeforeStartAttemptIds: [],
  outOfWindowAttemptIds: [],
  unknownDebitAttempts: 1,
  unknownDebitAttemptIds: ["attempt-2"],
  uncertainDebitCategories: {
    failed_after_start: ["attempt-2"],
    cancelled_after_start: [],
    unknown_acceptance: [],
    rejected_after_start: [],
    conflicting_models: [],
    unresolved_duplicate_identity: [],
    conflicting_duplicate_identity: [],
  },
  unclassifiedAttempts: 0,
  unclassifiedAttemptIds: [],
  unclassifiedReasons: {
    unknown_surface: [],
    unattributed_origin: [],
    unknown_ownership: [],
    ownership_mismatch: [],
    not_generation_started: [],
    unknown_model: [],
    ineligible_family: [],
    window_membership_unknown: [],
    conflicting_duplicate_identity: [],
  },
  windowMembershipUnknownAttemptIds: [],
  assessments: [
    {
      attemptId: "attempt-2",
      disposition: "uncertain_debit_excluded",
      selectedFamily: "astra_pro",
      basis: "requested",
      applicableBucketIds: ["bucket-1"],
      contributedBucketIds: [],
      membershipByBucket: {
        "bucket-1": "in",
      },
      uncertainDebitCategories: ["failed_after_start"],
      unclassifiedReasons: [],
    },
  ],
  label: "Working estimate; not an official remaining quota",
};

function input(
  observationContext?: ReportExportInput["context"]["quotaObservationContextByBucket"],
): ReportExportInput {
  return {
    report,
    quotaEstimate,
    context: {
      evaluatedAt: "2026-09-07T12:00:00.000Z",
      timezone: "America/New_York",
      freshness: {
        status: "fresh",
        observedAt: "2026-09-07T11:59:00.000Z",
        ageMs: 60_000,
        maxAgeMs: 3_600_000,
      },
      coverage: {
        status: "partial",
        source: "history-ledger",
        observedAt: "2026-09-07T12:00:00.000Z",
      },
      provenance: {
        report: {
          source: "local-ledger",
          revisionId: "revision-1",
          inputFingerprint: "fingerprint-1",
          mappingVersion: "mapping-1",
        },
        policy: {
          source: "local-policy",
          policyId: "policy-1",
          version: "policy-v1",
          provenance: "documented-seed",
        },
        windows: {
          "bucket-1": {
            source: "provider_explicit",
            provenance: "fixture-window",
            windowType: "provider_explicit",
            start: "2026-09-07T00:00:00.000Z",
            end: "2026-09-08T00:00:00.000Z",
            timezone: "UTC",
            evidenceObservedAt: "2026-09-07T11:58:00.000Z",
          },
        },
      },
      ...(observationContext === undefined
        ? {}
        : { quotaObservationContextByBucket: observationContext }),
    },
  };
}

describe("offline report exports", () => {
  it("should project allowlisted fields and preserve unknown observations and nulls", () => {
    const document = buildReportExport(input());
    const bucket = document.quotaEstimate?.buckets["bucket-1"];

    expect(bucket?.serverObservation).toMatchObject({
      source: "unknown",
      qualification: "unknown",
      confidence: "unknown",
      scoped: false,
    });
    expect(bucket?.workingRemainingEstimate).toBeNull();
    expect(bucket?.workingRemainingUnclamped).toBe(-2);
    expect(bucket?.discrepancy).toMatchObject({
      kind: "local_usage_exceeds_capacity",
      excess: 2,
      qualification: "diagnostic_only",
    });
    expect(document.provenance.report.inputFingerprint).toBe("fingerprint-1");
    expect(document.provenance.windows["bucket-1"]?.provenance).toBe(
      "fixture-window",
    );
    expect(JSON.stringify(document)).not.toContain("must-not-export");
  });

  it("should assign A only to explicitly provider-verified scoped observations", () => {
    const provider = buildReportExport(
      input({
        "bucket-1": {
          source: "provider",
          qualification: "provider_verified",
          provenance: "provider-window-observation",
          scoped: true,
        },
      }),
    );
    expect(provider.quotaEstimate?.buckets["bucket-1"]?.serverObservation).toMatchObject({
      source: "provider",
      qualification: "provider_verified",
      confidence: "A",
    });

    const manual = buildReportExport(
      input({
        "bucket-1": {
          source: "manual",
          qualification: "manual",
          provenance: "operator-note",
          scoped: true,
        },
      }),
    );
    expect(manual.quotaEstimate?.buckets["bucket-1"]?.serverObservation).toMatchObject({
      source: "manual",
      qualification: "manual",
      confidence: "F",
    });
    expect(renderMarkdown(input()).match(/Confidence: A/g)).toBeNull();
  });

  it("should escape CSV formula cells and Markdown table values", () => {
    const csv = renderCsv(input());
    expect(csv).toContain(`"'=SUM(A1:A2), ""quoted"""`);
    expect(csv).toContain("\"page|1\"");

    const markdown = renderMarkdown(input());
    expect(markdown).toContain("Relevance: 100% to requested interval");
    expect(markdown).toContain("model\\|pipe");
    expect(markdown).toContain("-2 (local\\_usage\\_exceeds\\_capacity");
    expect(markdown).toContain(
      "Working estimate; not an official remaining quota",
    );
  });

  it("should dispatch all supported renderers", () => {
    expect(JSON.parse(renderReport(input(), "json")).schemaVersion).toBe(
      "d1-752-report-export-v1",
    );
    expect(renderReport(input(), "csv").split("\r\n")[0]).toContain("section");
    expect(renderReport(input(), "markdown")).toContain("# Offline report export");
  });
});
