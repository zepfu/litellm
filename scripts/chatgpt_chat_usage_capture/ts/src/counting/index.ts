/**
 * Package-internal entrypoint for the preserved pure counting modules.
 *
 * This surface intentionally exposes in-memory reconstruction, model mapping,
 * quota estimation, and reset-window primitives only. It has no persistence,
 * browser, transport, CLI, or service coupling; the future bounded sidecar
 * worker owns the integration boundary.
 */

export * from "../contracts/records.js";
export * from "../contracts/history.js";
export * from "../ledger/identity.js";
export * from "../ledger/types.js";
export * from "../normalize/model-mapping.js";
export * from "../normalize/reconstruct.js";
export {
  WORKING_ESTIMATOR,
  WORKING_ESTIMATE_LABEL,
  SEED_CHAT_PRO_QUOTA_POLICY,
  selectEstimatedFamily,
  hasGenerationStartEvidence,
  classifyQuotaAttempt,
  estimateQuota,
} from "../accounting/quota.js";
export type {
  WorkingEstimator,
  QuotaCoverageStatus,
  UncertainAttemptMode,
  WindowMembership as QuotaWindowMembership,
  QuotaBucketKind,
  EstimateBasis,
  UncertainDebitCategory,
  UnclassifiedQuotaReason,
  QuotaAttemptDisposition,
  QuotaBucketPolicy,
  QuotaPolicy,
  ResolvedQuotaWindow,
  DirectServerQuotaObservation,
  QuotaOwnershipExpectation,
  QuotaEstimatorInput,
  QuotaAttemptAssessment,
  QuotaDiscrepancy,
  QuotaBucketEstimate,
  QuotaEstimate,
  EstimatedFamilySelection,
} from "../accounting/quota.js";
export * from "../accounting/windows.js";
