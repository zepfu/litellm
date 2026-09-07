import type {
  MappingEvidence,
  MappingRule,
  ModelMappingVersion,
} from "../ledger/types.js";

export const INITIAL_MODEL_MAPPING: ModelMappingVersion = {
  version: "initial-unmapped",
  canonicalFamilies: ["astra_pro", "sol_pro", "other_chat", "unknown"],
  rules: [],
  reviewStatus: "draft",
  source: "collector-empty-seed",
  createdAt: "2026-09-07T00:00:00.000Z",
};

export interface ModelSuggestion {
  slug: string;
  mode: string | null;
  reasoningEffort: string | null;
  observedAttempts: number;
  completedAnswers: number;
  suggestedFamily: string | null;
  reason: string;
}

export class ModelMappingError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ModelMappingError";
  }
}

export function validateMappingVersion(mapping: ModelMappingVersion): void {
  if (!mapping.version.trim()) {
    throw new ModelMappingError("mapping version must not be empty");
  }
  if (mapping.canonicalFamilies.length === 0) {
    throw new ModelMappingError("mapping must declare at least one family");
  }
  const families = new Set(mapping.canonicalFamilies);
  const selectors = new Map<string, string>();
  for (const rule of mapping.rules) {
    if (!rule.slug.trim()) {
      throw new ModelMappingError("mapping rule slug must not be empty");
    }
    if (!families.has(rule.family)) {
      throw new ModelMappingError(
        `mapping rule family is not canonical: ${rule.family}`,
      );
    }
    const selector = mappingSelector(rule);
    const previous = selectors.get(selector);
    if (previous !== undefined && previous !== rule.family) {
      throw new ModelMappingError(
        `conflicting mapping rules for selector ${selector}`,
      );
    }
    selectors.set(selector, rule.family);
  }
}

export function mapModelEvidence(
  evidence: MappingEvidence,
  mapping: ModelMappingVersion,
  collectorAccountId?: string,
): string | null {
  validateMappingVersion(mapping);
  const candidates = mapping.rules
    .filter((rule) => {
      if (
        rule.collectorAccountId !== undefined &&
        rule.collectorAccountId !== null &&
        rule.collectorAccountId !== collectorAccountId
      ) {
        return false;
      }
      return rule.slug === evidence.slug;
    })
    .filter((rule) => matchesOptional(rule.mode, evidence.mode))
    .filter((rule) => matchesOptional(rule.reasoningEffort, evidence.reasoningEffort))
    .sort((left, right) => ruleSpecificity(right, collectorAccountId) - ruleSpecificity(left, collectorAccountId));
  return candidates[0]?.family ?? null;
}

export function suggestModelMappings(
  rows: ReadonlyArray<{
    requestedModelRaw: string | null;
    requestedModeRaw: string | null;
    requestedReasoningEffortRaw: string | null;
    recordedFinalModelRaw: string | null;
    resolvedModelRaw: string | null;
    completedAnswer: boolean;
  }>,
  mapping: ModelMappingVersion,
  collectorAccountId?: string,
): ModelSuggestion[] {
  const grouped = new Map<string, ModelSuggestion>();
  for (const row of rows) {
    const candidateValues: MappingEvidence[] = [
      {
        slug: row.requestedModelRaw,
        mode: row.requestedModeRaw,
        reasoningEffort: row.requestedReasoningEffortRaw,
      },
      {
        slug: row.recordedFinalModelRaw,
        mode: null,
        reasoningEffort: null,
      },
      {
        slug: row.resolvedModelRaw,
        mode: null,
        reasoningEffort: null,
      },
    ];
    const values = new Map<string, MappingEvidence>();
    for (const value of candidateValues) {
      if (value.slug === null) {
        continue;
      }
      values.set(
        JSON.stringify([value.slug, value.mode, value.reasoningEffort]),
        value,
      );
    }
    for (const [key, value] of values) {
      if (value.slug === null) {
        continue;
      }
      const existing = grouped.get(key);
      const family = mapModelEvidence(value, mapping, collectorAccountId);
      if (existing) {
        existing.observedAttempts += 1;
        if (row.completedAnswer) {
          existing.completedAnswers += 1;
        }
        continue;
      }
      grouped.set(key, {
        slug: value.slug,
        mode: value.mode,
        reasoningEffort: value.reasoningEffort,
        observedAttempts: 1,
        completedAnswers: row.completedAnswer ? 1 : 0,
        suggestedFamily: family,
        reason: family
          ? "matches an existing reviewed rule"
          : "unmapped raw model evidence requires operator review",
      });
    }
  }
  return [...grouped.values()].sort((left, right) =>
    JSON.stringify([left.slug, left.mode, left.reasoningEffort]).localeCompare(
      JSON.stringify([right.slug, right.mode, right.reasoningEffort]),
    ),
  );
}

function mappingSelector(rule: MappingRule): string {
  return JSON.stringify([
    rule.collectorAccountId ?? null,
    rule.slug,
    rule.mode ?? null,
    rule.reasoningEffort ?? null,
  ]);
}

function matchesOptional(ruleValue: string | null | undefined, observed: string | null): boolean {
  return ruleValue === undefined || ruleValue === null || ruleValue === observed;
}

function ruleSpecificity(rule: MappingRule, collectorAccountId?: string): number {
  let score = 1;
  if (rule.mode !== undefined && rule.mode !== null) {
    score += 2;
  }
  if (rule.reasoningEffort !== undefined && rule.reasoningEffort !== null) {
    score += 2;
  }
  if (
    rule.collectorAccountId !== undefined &&
    rule.collectorAccountId !== null &&
    rule.collectorAccountId === collectorAccountId
  ) {
    score += 4;
  }
  if (rule.reviewed !== false) {
    score += 1;
  }
  return score;
}
