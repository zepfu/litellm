import type { ReconstructedAttempt } from "../ledger/types.js";

export const WORKING_ESTIMATOR = "requested_if_known_else_recorded_final" as const;
export const WORKING_ESTIMATE_LABEL =
  "Working estimate; not an official remaining quota";

export type WorkingEstimator = typeof WORKING_ESTIMATOR;
export type QuotaCoverageStatus = "complete" | "partial" | "unknown";
export type UncertainAttemptMode = "exclude" | "include";
export type WindowMembership = "in" | "out" | "ambiguous" | "unknown";
export type QuotaBucketKind = "individual" | "shared";
export type EstimateBasis = "requested" | "final_response_inference";

export type UncertainDebitCategory =
  | "failed_after_start"
  | "cancelled_after_start"
  | "unknown_acceptance"
  | "rejected_after_start"
  | "conflicting_models"
  | "unresolved_duplicate_identity";

export type UnclassifiedQuotaReason =
  | "unknown_surface"
  | "unattributed_origin"
  | "unknown_ownership"
  | "ownership_mismatch"
  | "not_generation_started"
  | "unknown_model"
  | "ineligible_family"
  | "window_membership_unknown";

export type QuotaAttemptDisposition =
  | "counted"
  | "uncertain_debit_excluded"
  | "rejected_before_start"
  | "out_of_window"
  | "out_of_scope"
  | "unclassified";

export interface QuotaBucketPolicy {
  bucketId: string;
  kind: QuotaBucketKind;
  eligibleFamilies: readonly string[];
  capacity: number;
}

export interface QuotaPolicy {
  policyId: string;
  version: string;
  buckets: readonly QuotaBucketPolicy[];
}

/**
 * Application-local seed policy from build-spec section 2.1. These values are
 * policy inputs for an estimate, not claims about provider charging behavior.
 */
export const SEED_CHAT_PRO_QUOTA_POLICY: QuotaPolicy = {
  policyId: "chat-pro-seed",
  version: "seed-2.1",
  buckets: [
    {
      bucketId: "astra_weekly",
      kind: "individual",
      eligibleFamilies: ["astra_pro"],
      capacity: 200,
    },
    {
      bucketId: "sol_daily",
      kind: "individual",
      eligibleFamilies: ["sol_pro"],
      capacity: 170,
    },
    {
      bucketId: "pro_combined_daily",
      kind: "shared",
      eligibleFamilies: ["astra_pro", "sol_pro"],
      capacity: 200,
    },
  ],
};

/**
 * Window resolution is intentionally outside this module. The caller supplies
 * the already-resolved membership for each attempt and bucket.
 */
export interface ResolvedQuotaWindow {
  bucketId: string;
  windowId: string | null;
  status: "known" | "unknown";
  membershipByAttemptId: Readonly<Record<string, WindowMembership>>;
}

/** A scoped observation supplied by a server/operator adapter. */
export interface DirectServerQuotaObservation {
  bucketId: string;
  windowId: string;
  remaining: number;
  observedAt: string;
}

export interface QuotaOwnershipExpectation {
  quotaOwnerId: string;
  providerUserId?: string | null;
  workspaceId?: string | null;
}

export interface QuotaEstimatorInput {
  attempts: ReadonlyArray<ReconstructedAttempt>;
  policy?: QuotaPolicy;
  windows: ReadonlyArray<ResolvedQuotaWindow>;
  coverage: QuotaCoverageStatus;
  ownership: QuotaOwnershipExpectation;
  serverObservations?: ReadonlyArray<DirectServerQuotaObservation>;
  uncertainAttemptMode?: UncertainAttemptMode;
}

export interface QuotaAttemptAssessment {
  attemptId: string;
  disposition: QuotaAttemptDisposition;
  selectedFamily: string | null;
  basis: EstimateBasis | null;
  applicableBucketIds: string[];
  contributedBucketIds: string[];
  membershipByBucket: Record<string, WindowMembership>;
  uncertainDebitCategories: UncertainDebitCategory[];
  unclassifiedReasons: UnclassifiedQuotaReason[];
}

export interface QuotaDiscrepancy {
  kind: "local_usage_exceeds_capacity";
  excess: number;
}

export interface QuotaBucketEstimate {
  bucketId: string;
  kind: QuotaBucketKind;
  eligibleFamilies: string[];
  capacity: number;
  windowId: string | null;
  workingUsageEstimate: number | null;
  workingRemainingEstimate: number | null;
  workingRemainingUnclamped: number | null;
  discrepancy: QuotaDiscrepancy | null;
  serverReportedRemaining: DirectServerQuotaObservation | null;
  countedAttemptIds: string[];
  unknownDebitAttemptIds: string[];
  unknownMembershipAttemptIds: string[];
}

export interface QuotaEstimate {
  policyId: string;
  policyVersion: string;
  workingEstimator: WorkingEstimator;
  uncertainAttemptMode: UncertainAttemptMode;
  coverage: QuotaCoverageStatus;
  buckets: Record<string, QuotaBucketEstimate>;
  workingUsageEstimateByBucket: Record<string, number | null>;
  workingRemainingEstimateByBucket: Record<string, number | null>;
  workingRemainingUnclampedByBucket: Record<string, number | null>;
  serverReportedRemainingByBucket: Record<
    string,
    DirectServerQuotaObservation | null
  >;
  workingHeadroomByFamily: Record<string, number | null>;
  countedAttemptIds: string[];
  rejectedBeforeStartAttemptIds: string[];
  outOfWindowAttemptIds: string[];
  unknownDebitAttempts: number;
  unknownDebitAttemptIds: string[];
  uncertainDebitCategories: Record<UncertainDebitCategory, string[]>;
  unclassifiedAttempts: number;
  unclassifiedAttemptIds: string[];
  unclassifiedReasons: Record<UnclassifiedQuotaReason, string[]>;
  windowMembershipUnknownAttemptIds: string[];
  assessments: QuotaAttemptAssessment[];
  label: string;
}

const UNCERTAIN_DEBIT_CATEGORIES: readonly UncertainDebitCategory[] = [
  "failed_after_start",
  "cancelled_after_start",
  "unknown_acceptance",
  "rejected_after_start",
  "conflicting_models",
  "unresolved_duplicate_identity",
];

const UNCLASSIFIED_REASONS: readonly UnclassifiedQuotaReason[] = [
  "unknown_surface",
  "unattributed_origin",
  "unknown_ownership",
  "ownership_mismatch",
  "not_generation_started",
  "unknown_model",
  "ineligible_family",
  "window_membership_unknown",
];

export interface EstimatedFamilySelection {
  family: string | null;
  basis: EstimateBasis | null;
}

interface MutableBucketEstimate {
  policy: QuotaBucketPolicy;
  window: ResolvedQuotaWindow | null;
  usage: number | null;
  membershipComplete: boolean;
  countedAttemptIds: Set<string>;
  unknownDebitAttemptIds: Set<string>;
  unknownMembershipAttemptIds: Set<string>;
}

export function selectEstimatedFamily(
  attempt: ReconstructedAttempt,
): EstimatedFamilySelection {
  const requestedFamily = nonEmptyString(attempt.requestedFamily);
  if (requestedFamily !== null) {
    return { family: requestedFamily, basis: "requested" };
  }
  if (attempt.completedAnswer) {
    const recordedFinalFamily = nonEmptyString(attempt.recordedFinalFamily);
    if (recordedFinalFamily !== null) {
      return {
        family: recordedFinalFamily,
        basis: "final_response_inference",
      };
    }
  }
  return { family: null, basis: null };
}

export function hasGenerationStartEvidence(
  attempt: ReconstructedAttempt,
): boolean {
  return attempt.generationStarted || attempt.completedAnswer;
}

export function classifyQuotaAttempt(
  attempt: ReconstructedAttempt,
  options: {
    policy?: QuotaPolicy;
    windows: ReadonlyArray<ResolvedQuotaWindow>;
    ownership: QuotaOwnershipExpectation;
    uncertainAttemptMode?: UncertainAttemptMode;
  },
): QuotaAttemptAssessment {
  const policy = options.policy ?? SEED_CHAT_PRO_QUOTA_POLICY;
  validatePolicy(policy);
  const windows = indexWindows(options.windows, policy);
  validateOwnership(options.ownership);
  const uncertainAttemptMode = options.uncertainAttemptMode ?? "exclude";
  return classifyQuotaAttemptWithIndexes(
    attempt,
    policy,
    windows,
    options.ownership,
    uncertainAttemptMode,
  );
}

function classifyQuotaAttemptWithIndexes(
  attempt: ReconstructedAttempt,
  policy: QuotaPolicy,
  windows: ReadonlyMap<string, ResolvedQuotaWindow>,
  ownership: QuotaOwnershipExpectation,
  uncertainAttemptMode: UncertainAttemptMode,
): QuotaAttemptAssessment {
  const selected = selectEstimatedFamily(attempt);
  const emptyMembership: Record<string, WindowMembership> = {};

  const scopeReason = scopeReasonFor(attempt, ownership);
  if (scopeReason !== null) {
    return {
      attemptId: attempt.attemptId,
      disposition: "out_of_scope",
      selectedFamily: selected.family,
      basis: selected.basis,
      applicableBucketIds: [],
      contributedBucketIds: [],
      membershipByBucket: emptyMembership,
      uncertainDebitCategories: [],
      unclassifiedReasons: [scopeReason],
    };
  }

  if (attempt.outcome === "rejected_before_start") {
    return {
      attemptId: attempt.attemptId,
      disposition: "rejected_before_start",
      selectedFamily: selected.family,
      basis: selected.basis,
      applicableBucketIds: [],
      contributedBucketIds: [],
      membershipByBucket: emptyMembership,
      uncertainDebitCategories: [],
      unclassifiedReasons: [],
    };
  }

  if (!hasGenerationStartEvidence(attempt)) {
    return {
      attemptId: attempt.attemptId,
      disposition: "unclassified",
      selectedFamily: selected.family,
      basis: selected.basis,
      applicableBucketIds: [],
      contributedBucketIds: [],
      membershipByBucket: emptyMembership,
      uncertainDebitCategories: [],
      unclassifiedReasons: ["not_generation_started"],
    };
  }

  if (selected.family === null) {
    return {
      attemptId: attempt.attemptId,
      disposition: "unclassified",
      selectedFamily: null,
      basis: null,
      applicableBucketIds: [],
      contributedBucketIds: [],
      membershipByBucket: emptyMembership,
      uncertainDebitCategories: [],
      unclassifiedReasons: ["unknown_model"],
    };
  }

  const selectedFamily = selected.family;
  const applicableBuckets = policy.buckets.filter((bucket) =>
    bucket.eligibleFamilies.includes(selectedFamily),
  );
  if (applicableBuckets.length === 0) {
    return {
      attemptId: attempt.attemptId,
      disposition: "unclassified",
      selectedFamily: selected.family,
      basis: selected.basis,
      applicableBucketIds: [],
      contributedBucketIds: [],
      membershipByBucket: emptyMembership,
      uncertainDebitCategories: [],
      unclassifiedReasons: ["ineligible_family"],
    };
  }

  const membershipByBucket = Object.fromEntries(
    applicableBuckets
      .map((bucket) => ({
        bucketId: bucket.bucketId,
        membership: membershipFor(attempt.attemptId, bucket.bucketId, windows),
      }))
      .sort((left, right) => left.bucketId.localeCompare(right.bucketId))
      .map((entry) => [entry.bucketId, entry.membership]),
  ) as Record<string, WindowMembership>;
  const contributedBucketIds = applicableBuckets
    .filter((bucket) => membershipByBucket[bucket.bucketId] === "in")
    .map((bucket) => bucket.bucketId)
    .sort();
  const hasUncertainMembership = applicableBuckets.some((bucket) =>
    isUncertainMembership(membershipByBucket[bucket.bucketId]),
  );
  const uncertainDebitCategories = uncertainDebitCategoriesFor(attempt);
  const hasCandidateMembership =
    contributedBucketIds.length > 0 || hasUncertainMembership;

  if (!hasCandidateMembership) {
    return {
      attemptId: attempt.attemptId,
      disposition: "out_of_window",
      selectedFamily: selected.family,
      basis: selected.basis,
      applicableBucketIds: applicableBuckets
        .map((bucket) => bucket.bucketId)
        .sort(),
      contributedBucketIds: [],
      membershipByBucket,
      uncertainDebitCategories: [],
      unclassifiedReasons: [],
    };
  }

  if (
    uncertainDebitCategories.length > 0 &&
    uncertainAttemptMode === "exclude"
  ) {
    return {
      attemptId: attempt.attemptId,
      disposition: "uncertain_debit_excluded",
      selectedFamily: selected.family,
      basis: selected.basis,
      applicableBucketIds: applicableBuckets
        .map((bucket) => bucket.bucketId)
        .sort(),
      contributedBucketIds: [],
      membershipByBucket,
      uncertainDebitCategories,
      unclassifiedReasons: hasUncertainMembership
        ? ["window_membership_unknown"]
        : [],
    };
  }

  if (contributedBucketIds.length === 0) {
    return {
      attemptId: attempt.attemptId,
      disposition: "unclassified",
      selectedFamily: selected.family,
      basis: selected.basis,
      applicableBucketIds: applicableBuckets
        .map((bucket) => bucket.bucketId)
        .sort(),
      contributedBucketIds: [],
      membershipByBucket,
      uncertainDebitCategories,
      unclassifiedReasons: ["window_membership_unknown"],
    };
  }

  return {
    attemptId: attempt.attemptId,
    disposition: "counted",
    selectedFamily: selected.family,
    basis: selected.basis,
    applicableBucketIds: applicableBuckets
      .map((bucket) => bucket.bucketId)
      .sort(),
    contributedBucketIds,
    membershipByBucket,
    uncertainDebitCategories,
    unclassifiedReasons: hasUncertainMembership
      ? ["window_membership_unknown"]
      : [],
  };
}

export function estimateQuota(input: QuotaEstimatorInput): QuotaEstimate {
  const policy = input.policy ?? SEED_CHAT_PRO_QUOTA_POLICY;
  validatePolicy(policy);
  validateCoverage(input.coverage);
  validateOwnership(input.ownership);
  const windows = indexWindows(input.windows, policy);
  const serverObservations = indexServerObservations(
    input.serverObservations ?? [],
    policy,
  );
  const uncertainAttemptMode = input.uncertainAttemptMode ?? "exclude";
  const attempts = [...input.attempts].sort((left, right) =>
    left.attemptId.localeCompare(right.attemptId),
  );
  assertUniqueAttemptIds(attempts);

  const bucketStates = new Map<string, MutableBucketEstimate>();
  for (const bucket of policy.buckets) {
    const window = windows.get(bucket.bucketId) ?? null;
    bucketStates.set(bucket.bucketId, {
      policy: bucket,
      window,
      usage: window?.status === "known" ? 0 : null,
      membershipComplete: window?.status === "known",
      countedAttemptIds: new Set<string>(),
      unknownDebitAttemptIds: new Set<string>(),
      unknownMembershipAttemptIds: new Set<string>(),
    });
  }

  const uncertainDebitIds = new Set<string>();
  const uncertainDebitByCategory = makeCategorySets(UNCERTAIN_DEBIT_CATEGORIES);
  const unclassifiedIds = new Set<string>();
  const unclassifiedByReason = makeCategorySets(UNCLASSIFIED_REASONS);
  const countedAttemptIds = new Set<string>();
  const rejectedBeforeStartAttemptIds = new Set<string>();
  const outOfWindowAttemptIds = new Set<string>();
  const windowMembershipUnknownAttemptIds = new Set<string>();
  const assessments: QuotaAttemptAssessment[] = [];

  for (const attempt of attempts) {
    const assessment = classifyQuotaAttemptWithIndexes(
      attempt,
      policy,
      windows,
      input.ownership,
      uncertainAttemptMode,
    );
    assessments.push(assessment);

    if (assessment.disposition === "counted") {
      countedAttemptIds.add(assessment.attemptId);
    }
    if (assessment.disposition === "rejected_before_start") {
      rejectedBeforeStartAttemptIds.add(assessment.attemptId);
    }
    if (assessment.disposition === "out_of_window") {
      outOfWindowAttemptIds.add(assessment.attemptId);
    }
    if (
      assessment.disposition === "out_of_scope" ||
      assessment.disposition === "unclassified" ||
      assessment.unclassifiedReasons.includes("window_membership_unknown")
    ) {
      unclassifiedIds.add(assessment.attemptId);
    }
    for (const reason of assessment.unclassifiedReasons) {
      unclassifiedByReason.get(reason)?.add(assessment.attemptId);
      if (reason === "window_membership_unknown") {
        windowMembershipUnknownAttemptIds.add(assessment.attemptId);
      }
    }
    for (const category of assessment.uncertainDebitCategories) {
      uncertainDebitByCategory.get(category)?.add(assessment.attemptId);
    }
    if (assessment.uncertainDebitCategories.length > 0) {
      uncertainDebitIds.add(assessment.attemptId);
    }

    for (const bucketId of assessment.applicableBucketIds) {
      const state = bucketStates.get(bucketId);
      if (state === undefined) {
        throw new Error(`missing bucket state for ${bucketId}`);
      }
      const membership = assessment.membershipByBucket[bucketId];
      if (membership === undefined) {
        continue;
      }
      if (isUncertainMembership(membership)) {
        state.membershipComplete = false;
        state.unknownMembershipAttemptIds.add(assessment.attemptId);
        windowMembershipUnknownAttemptIds.add(assessment.attemptId);
      }
      if (assessment.uncertainDebitCategories.length > 0) {
        if (membership === "in" || isUncertainMembership(membership)) {
          state.unknownDebitAttemptIds.add(assessment.attemptId);
        }
      }
    }

    for (const bucketId of assessment.contributedBucketIds) {
      const state = bucketStates.get(bucketId);
      if (state === undefined) {
        throw new Error(`missing bucket state for ${bucketId}`);
      }
      if (state.usage !== null) {
        state.usage += 1;
      }
      state.countedAttemptIds.add(assessment.attemptId);
    }
  }

  const bucketEstimates = Object.fromEntries(
    policy.buckets
      .map((bucket) => {
        const state = bucketStates.get(bucket.bucketId);
        if (state === undefined) {
          throw new Error(`missing bucket state for ${bucket.bucketId}`);
        }
        return [bucket.bucketId, buildBucketEstimate(state, input.coverage, serverObservations)] as const;
      })
      .sort(([left], [right]) => left.localeCompare(right)),
  ) as Record<string, QuotaBucketEstimate>;

  const workingUsageEstimateByBucket = mapBucketValues(
    bucketEstimates,
    (bucket) => bucket.workingUsageEstimate,
  );
  const workingRemainingEstimateByBucket = mapBucketValues(
    bucketEstimates,
    (bucket) => bucket.workingRemainingEstimate,
  );
  const workingRemainingUnclampedByBucket = mapBucketValues(
    bucketEstimates,
    (bucket) => bucket.workingRemainingUnclamped,
  );
  const serverReportedRemainingByBucket = mapBucketValues(
    bucketEstimates,
    (bucket) => bucket.serverReportedRemaining,
  );

  return {
    policyId: policy.policyId,
    policyVersion: policy.version,
    workingEstimator: WORKING_ESTIMATOR,
    uncertainAttemptMode,
    coverage: input.coverage,
    buckets: bucketEstimates,
    workingUsageEstimateByBucket,
    workingRemainingEstimateByBucket,
    workingRemainingUnclampedByBucket,
    serverReportedRemainingByBucket,
    workingHeadroomByFamily: buildHeadroomByFamily(policy, bucketEstimates),
    countedAttemptIds: [...countedAttemptIds].sort(),
    rejectedBeforeStartAttemptIds: [...rejectedBeforeStartAttemptIds].sort(),
    outOfWindowAttemptIds: [...outOfWindowAttemptIds].sort(),
    unknownDebitAttempts: uncertainDebitIds.size,
    unknownDebitAttemptIds: [...uncertainDebitIds].sort(),
    uncertainDebitCategories: materializeCategorySets(
      uncertainDebitByCategory,
      UNCERTAIN_DEBIT_CATEGORIES,
    ),
    unclassifiedAttempts: unclassifiedIds.size,
    unclassifiedAttemptIds: [...unclassifiedIds].sort(),
    unclassifiedReasons: materializeCategorySets(
      unclassifiedByReason,
      UNCLASSIFIED_REASONS,
    ),
    windowMembershipUnknownAttemptIds: [
      ...windowMembershipUnknownAttemptIds,
    ].sort(),
    assessments,
    label: WORKING_ESTIMATE_LABEL,
  };
}

function buildBucketEstimate(
  state: MutableBucketEstimate,
  coverage: QuotaCoverageStatus,
  serverObservations: ReadonlyMap<string, DirectServerQuotaObservation>,
): QuotaBucketEstimate {
  const workingUsageEstimate = state.window?.status === "known"
    ? state.usage
    : null;
  const remainderKnown =
    state.window?.status === "known" &&
    coverage === "complete" &&
    state.membershipComplete &&
    workingUsageEstimate !== null;
  const workingRemainingUnclamped = remainderKnown
    ? state.policy.capacity - (workingUsageEstimate ?? 0)
    : null;
  const workingRemainingEstimate = workingRemainingUnclamped === null
    ? null
    : Math.max(0, workingRemainingUnclamped);
  const discrepancy =
    workingRemainingUnclamped !== null && workingRemainingUnclamped < 0
      ? {
          kind: "local_usage_exceeds_capacity" as const,
          excess: Math.abs(workingRemainingUnclamped),
        }
      : null;

  return {
    bucketId: state.policy.bucketId,
    kind: state.policy.kind,
    eligibleFamilies: [...state.policy.eligibleFamilies].sort(),
    capacity: state.policy.capacity,
    windowId: state.window?.status === "known" ? state.window.windowId : null,
    workingUsageEstimate,
    workingRemainingEstimate,
    workingRemainingUnclamped,
    discrepancy,
    serverReportedRemaining:
      serverObservations.get(state.policy.bucketId) ?? null,
    countedAttemptIds: [...state.countedAttemptIds].sort(),
    unknownDebitAttemptIds: [...state.unknownDebitAttemptIds].sort(),
    unknownMembershipAttemptIds: [
      ...state.unknownMembershipAttemptIds,
    ].sort(),
  };
}

function buildHeadroomByFamily(
  policy: QuotaPolicy,
  bucketEstimates: Readonly<Record<string, QuotaBucketEstimate>>,
): Record<string, number | null> {
  const families = new Set<string>();
  for (const bucket of policy.buckets) {
    for (const family of bucket.eligibleFamilies) {
      families.add(family);
    }
  }

  return Object.fromEntries(
    [...families]
      .sort()
      .map((family) => {
        const relevantBuckets = policy.buckets.filter((bucket) =>
          bucket.eligibleFamilies.includes(family),
        );
        const remainders = relevantBuckets.map(
          (bucket) => bucketEstimates[bucket.bucketId]?.workingRemainingEstimate,
        );
        const compatible =
          remainders.length > 0 &&
          remainders.every(
            (remaining): remaining is number =>
              remaining !== null && Number.isFinite(remaining),
          );
        return [family, compatible ? Math.min(...remainders) : null] as const;
      }),
  ) as Record<string, number | null>;
}

function mapBucketValues<T>(
  buckets: Readonly<Record<string, QuotaBucketEstimate>>,
  value: (bucket: QuotaBucketEstimate) => T,
): Record<string, T> {
  return Object.fromEntries(
    Object.keys(buckets)
      .sort()
      .map((bucketId) => [bucketId, value(buckets[bucketId] as QuotaBucketEstimate)]),
  ) as Record<string, T>;
}

function scopeReasonFor(
  attempt: ReconstructedAttempt,
  ownership: QuotaOwnershipExpectation,
): UnclassifiedQuotaReason | null {
  if (attempt.surface !== "chat" || attempt.scope.surface !== "chat") {
    return "unknown_surface";
  }
  const origin = attempt.origin?.toLowerCase() ?? null;
  if (
    origin === "shared" ||
    origin === "imported" ||
    origin === "copied" ||
    origin === "true"
  ) {
    return "unattributed_origin";
  }
  if (attempt.scope.quotaOwnerId === null) {
    return "unknown_ownership";
  }
  if (attempt.scope.quotaOwnerId !== ownership.quotaOwnerId) {
    return "ownership_mismatch";
  }
  if (
    ownership.providerUserId !== undefined &&
    attempt.scope.providerUserId !== ownership.providerUserId
  ) {
    return "ownership_mismatch";
  }
  if (
    ownership.workspaceId !== undefined &&
    attempt.scope.workspaceId !== ownership.workspaceId
  ) {
    return "ownership_mismatch";
  }
  return null;
}

function uncertainDebitCategoriesFor(
  attempt: ReconstructedAttempt,
): UncertainDebitCategory[] {
  const categories = new Set<UncertainDebitCategory>();
  switch (attempt.outcome) {
    case "failed_after_start":
      categories.add("failed_after_start");
      break;
    case "cancelled_after_start":
      categories.add("cancelled_after_start");
      break;
    case "completion_unknown":
      categories.add("unknown_acceptance");
      break;
    case "rejected_after_start":
      categories.add("rejected_after_start");
      break;
    default:
      break;
  }
  if (
    nonEmptyString(attempt.requestedFamily) !== null &&
    nonEmptyString(attempt.recordedFinalFamily) !== null &&
    attempt.requestedFamily !== attempt.recordedFinalFamily
  ) {
    categories.add("conflicting_models");
  }
  if (
    attempt.identityBasis === "unresolved" ||
    attempt.warnings.includes("unresolved_linkage")
  ) {
    categories.add("unresolved_duplicate_identity");
  }
  return UNCERTAIN_DEBIT_CATEGORIES.filter((category) =>
    categories.has(category),
  );
}

function membershipFor(
  attemptId: string,
  bucketId: string,
  windows: ReadonlyMap<string, ResolvedQuotaWindow>,
): WindowMembership {
  const window = windows.get(bucketId);
  if (window === undefined || window.status === "unknown") {
    return "unknown";
  }
  return window.membershipByAttemptId[attemptId] ?? "unknown";
}

function isUncertainMembership(
  membership: WindowMembership | undefined,
): boolean {
  return membership === "ambiguous" || membership === "unknown";
}

function indexWindows(
  windows: ReadonlyArray<ResolvedQuotaWindow>,
  policy: QuotaPolicy,
): Map<string, ResolvedQuotaWindow> {
  const policyBucketIds = new Set(policy.buckets.map((bucket) => bucket.bucketId));
  const indexed = new Map<string, ResolvedQuotaWindow>();
  for (const window of windows) {
    if (!policyBucketIds.has(window.bucketId)) {
      throw new Error(`window references unknown bucket ${window.bucketId}`);
    }
    if (indexed.has(window.bucketId)) {
      throw new Error(`duplicate resolved window for ${window.bucketId}`);
    }
    if (window.status === "known" && nonEmptyString(window.windowId) === null) {
      throw new Error(`known window ${window.bucketId} requires windowId`);
    }
    for (const membership of Object.values(window.membershipByAttemptId)) {
      if (
        membership !== "in" &&
        membership !== "out" &&
        membership !== "ambiguous" &&
        membership !== "unknown"
      ) {
        throw new Error(`invalid window membership for ${window.bucketId}`);
      }
    }
    indexed.set(window.bucketId, window);
  }
  return indexed;
}

function indexServerObservations(
  observations: ReadonlyArray<DirectServerQuotaObservation>,
  policy: QuotaPolicy,
): Map<string, DirectServerQuotaObservation> {
  const policyBucketIds = new Set(policy.buckets.map((bucket) => bucket.bucketId));
  const indexed = new Map<string, DirectServerQuotaObservation>();
  for (const observation of observations) {
    if (!policyBucketIds.has(observation.bucketId)) {
      throw new Error(
        `server observation references unknown bucket ${observation.bucketId}`,
      );
    }
    if (indexed.has(observation.bucketId)) {
      throw new Error(
        `duplicate direct server observation for ${observation.bucketId}`,
      );
    }
    if (!Number.isFinite(observation.remaining)) {
      throw new Error(
        `server observation remaining must be finite for ${observation.bucketId}`,
      );
    }
    if (nonEmptyString(observation.windowId) === null) {
      throw new Error(
        `server observation ${observation.bucketId} requires windowId`,
      );
    }
    indexed.set(observation.bucketId, observation);
  }
  return indexed;
}

function validatePolicy(policy: QuotaPolicy): void {
  if (nonEmptyString(policy.policyId) === null) {
    throw new Error("quota policy requires policyId");
  }
  if (nonEmptyString(policy.version) === null) {
    throw new Error("quota policy requires version");
  }
  const bucketIds = new Set<string>();
  for (const bucket of policy.buckets) {
    if (bucketIds.has(bucket.bucketId)) {
      throw new Error(`duplicate quota bucket ${bucket.bucketId}`);
    }
    bucketIds.add(bucket.bucketId);
    if (nonEmptyString(bucket.bucketId) === null) {
      throw new Error("quota bucket requires bucketId");
    }
    if (
      !Number.isFinite(bucket.capacity) ||
      bucket.capacity < 0
    ) {
      throw new Error(`invalid quota capacity for ${bucket.bucketId}`);
    }
    if (bucket.eligibleFamilies.length === 0) {
      throw new Error(`quota bucket ${bucket.bucketId} has no eligible family`);
    }
    const families = new Set<string>();
    for (const family of bucket.eligibleFamilies) {
      if (nonEmptyString(family) === null || families.has(family)) {
        throw new Error(`invalid eligible family for ${bucket.bucketId}`);
      }
      families.add(family);
    }
  }
  if (policy.buckets.length === 0) {
    throw new Error("quota policy requires at least one bucket");
  }
}

function validateCoverage(coverage: QuotaCoverageStatus): void {
  if (
    coverage !== "complete" &&
    coverage !== "partial" &&
    coverage !== "unknown"
  ) {
    throw new Error(`invalid quota coverage ${coverage}`);
  }
}

function validateOwnership(ownership: QuotaOwnershipExpectation): void {
  if (nonEmptyString(ownership.quotaOwnerId) === null) {
    throw new Error("quota ownership requires quotaOwnerId");
  }
}

function assertUniqueAttemptIds(
  attempts: ReadonlyArray<ReconstructedAttempt>,
): void {
  const ids = new Set<string>();
  for (const attempt of attempts) {
    if (ids.has(attempt.attemptId)) {
      throw new Error(`duplicate quota attempt ${attempt.attemptId}`);
    }
    ids.add(attempt.attemptId);
  }
}

function makeCategorySets<T extends string>(
  categories: readonly T[],
): Map<T, Set<string>> {
  return new Map(categories.map((category) => [category, new Set<string>()]));
}

function materializeCategorySets<T extends string>(
  values: ReadonlyMap<T, ReadonlySet<string>>,
  categories: readonly T[],
): Record<T, string[]> {
  return Object.fromEntries(
    categories.map((category) => [
      category,
      [...(values.get(category) ?? new Set<string>())].sort(),
    ]),
  ) as Record<T, string[]>;
}

function nonEmptyString(value: string | null): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}
