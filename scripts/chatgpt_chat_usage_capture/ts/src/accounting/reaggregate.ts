import type { ModelMappingVersion, LedgerScope } from "../ledger/types.js";
import type { Ledger } from "../ledger/store.js";
import { buildRawModelReport, type RawModelReport } from "./raw-model.js";

export interface ReaggregateResult {
  mappingVersion: string;
  evaluatedAt: string;
  rebuiltAttempts: number;
  reclassifiedAttempts: number;
  report: RawModelReport;
  revisionId: string;
  inputFingerprint: string;
}

export function reaggregate(
  ledger: Ledger,
  scope: LedgerScope,
  mapping: ModelMappingVersion,
  options: { evaluatedAt?: string; apply?: boolean } = {},
): ReaggregateResult {
  const evaluatedAt = options.evaluatedAt ?? new Date().toISOString();
  const run = (): ReaggregateResult => {
    const priorMappings = new Map(
      ledger
        .listAttempts(scope, true)
        .map((attempt) => [String(attempt.attemptId), String(attempt.mappingVersion)]),
    );
    const rebuiltAttempts = ledger.rebuildAttemptsFromMessages(
      scope,
      mapping,
      evaluatedAt,
    );
    const reclassifiedAttempts = ledger.reclassifyAttempts(
      scope,
      mapping,
      evaluatedAt,
    );
    const mappingChanges = ledger
      .listAttempts(scope, true)
      .filter(
        (attempt) =>
          priorMappings.get(String(attempt.attemptId)) !== undefined &&
          priorMappings.get(String(attempt.attemptId)) !== String(attempt.mappingVersion),
      ).length;
    const report = buildRawModelReport(ledger, scope, {
      durationMs: 7 * 24 * 60 * 60 * 1000,
      now: evaluatedAt,
    });
    const aggregate = ledger.writeAggregateRevision(scope, mapping.version, evaluatedAt, {
      mapping_version: mapping.version,
      rebuilt_attempts: rebuiltAttempts,
      reclassified_attempts: Math.max(reclassifiedAttempts, mappingChanges),
      raw_model_report: report,
    });
    return {
      mappingVersion: mapping.version,
      evaluatedAt,
      rebuiltAttempts,
      reclassifiedAttempts: Math.max(reclassifiedAttempts, mappingChanges),
      report,
      revisionId: aggregate.revisionId,
      inputFingerprint: aggregate.inputFingerprint,
    };
  };
  if (options.apply !== false) {
    return ledger.transaction(run);
  }
  try {
    ledger.transaction(() => {
      const result = run();
      throw new PreviewRollback(result);
    });
  } catch (error) {
    if (error instanceof PreviewRollback) {
      return error.result;
    }
    throw error;
  }
  throw new Error("reaggregate preview did not produce a result");
}

class PreviewRollback extends Error {
  constructor(readonly result: ReaggregateResult) {
    super("preview rollback");
    this.name = "PreviewRollback";
  }
}
