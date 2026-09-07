import type {
  MappingEvidence,
  MappingResolution,
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
  changeKind: "prospective",
  validFrom: null,
  validUntil: null,
  publishedAt: null,
  supersedesVersion: null,
  correctionOfVersion: null,
  provenance: {},
  warnings: [],
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

export function normalizeMappingVersion(
  mapping: ModelMappingVersion,
): ModelMappingVersion {
  const published =
    mapping.reviewStatus === "approved" || mapping.reviewStatus === "retired";
  return {
    ...mapping,
    changeKind: mapping.changeKind ?? "prospective",
    validFrom: mapping.validFrom ?? null,
    validUntil: mapping.validUntil ?? null,
    publishedAt:
      mapping.publishedAt ?? (published ? mapping.reviewedAt ?? mapping.createdAt : null),
    supersedesVersion: mapping.supersedesVersion ?? null,
    correctionOfVersion: mapping.correctionOfVersion ?? null,
    provenance: mapping.provenance ?? {},
    warnings: [...(mapping.warnings ?? [])],
  };
}

export function isMappingPublished(mapping: ModelMappingVersion): boolean {
  return mapping.reviewStatus === "approved" || mapping.reviewStatus === "retired";
}

export function validateMappingVersion(mapping: ModelMappingVersion): void {
  const normalized = normalizeMappingVersion(mapping);
  if (!normalized.version.trim()) {
    throw new ModelMappingError("mapping version must not be empty");
  }
  if (!normalized.source.trim()) {
    throw new ModelMappingError("mapping source must not be empty");
  }
  if (!isValidInstant(normalized.createdAt)) {
    throw new ModelMappingError("mapping createdAt must be a valid ISO timestamp");
  }
  if (normalized.canonicalFamilies.length === 0) {
    throw new ModelMappingError("mapping must declare at least one family");
  }
  const families = new Set<string>();
  for (const family of normalized.canonicalFamilies) {
    if (!family.trim() || families.has(family)) {
      throw new ModelMappingError(`invalid or duplicate canonical family: ${family}`);
    }
    families.add(family);
  }

  validateLifecycle(normalized);

  for (const rule of normalized.rules) {
    if (!rule.slug.trim()) {
      throw new ModelMappingError("mapping rule slug must not be empty");
    }
    if (
      rule.collectorAccountId !== undefined &&
      rule.collectorAccountId !== null &&
      !rule.collectorAccountId.trim()
    ) {
      throw new ModelMappingError("mapping rule account override must not be empty");
    }
    if (!families.has(rule.family)) {
      throw new ModelMappingError(
        `mapping rule family is not canonical: ${rule.family}`,
      );
    }
    if (isMappingPublished(normalized) && rule.reviewed !== true) {
      throw new ModelMappingError(
        `published mapping rule is not reviewed: ${mappingSelector(rule)}`,
      );
    }
  }

  for (let leftIndex = 0; leftIndex < normalized.rules.length; leftIndex += 1) {
    const left = normalized.rules[leftIndex]!;
    for (
      let rightIndex = leftIndex + 1;
      rightIndex < normalized.rules.length;
      rightIndex += 1
    ) {
      const right = normalized.rules[rightIndex]!;
      if (left.family === right.family || !rulesOverlap(left, right)) {
        continue;
      }
      if (compareRulePrecedence(left, right) === 0) {
        throw new ModelMappingError(
          `ambiguous mapping rule overlap: ${mappingSelector(left)} vs ${mappingSelector(right)}`,
        );
      }
    }
  }
}

export function resolveModelEvidence(
  evidence: MappingEvidence,
  mapping: ModelMappingVersion,
  collectorAccountId?: string,
  options: { at?: string | null } = {},
): MappingResolution {
  const normalized = normalizeMappingVersion(mapping);
  validateMappingVersion(normalized);

  if (!isMappingPublished(normalized)) {
    return {
      family: null,
      rule: null,
      warnings: ["mapping_unreviewed"],
      applied: false,
    };
  }

  const applicabilityWarnings = mappingApplicabilityWarnings(
    normalized,
    options.at ?? null,
  );
  if (applicabilityWarnings.length > 0) {
    return {
      family: null,
      rule: null,
      warnings: applicabilityWarnings,
      applied: false,
    };
  }

  const candidates = normalized.rules
    .filter((rule) => rule.reviewed === true)
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
    .sort((left, right) => compareRulePrecedence(right, left));

  const best = candidates[0] ?? null;
  if (best === null) {
    return {
      family: null,
      rule: null,
      warnings: [],
      applied: true,
    };
  }

  const tied = candidates.filter(
    (candidate) => compareRulePrecedence(candidate, best) === 0,
  );
  const tiedFamilies = new Set(tied.map((candidate) => candidate.family));
  if (tiedFamilies.size > 1) {
    return {
      family: null,
      rule: null,
      warnings: ["ambiguous_mapping_overlap"],
      applied: false,
    };
  }

  return {
    family: best.family,
    rule: best,
    warnings: [],
    applied: true,
  };
}

export function mapModelEvidence(
  evidence: MappingEvidence,
  mapping: ModelMappingVersion,
  collectorAccountId?: string,
  options: { at?: string | null } = {},
): string | null {
  return resolveModelEvidence(evidence, mapping, collectorAccountId, options).family;
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

function validateLifecycle(mapping: ModelMappingVersion): void {
  const changeKind = mapping.changeKind;
  if (changeKind !== "prospective" && changeKind !== "historical_correction") {
    throw new ModelMappingError(`unsupported mapping change kind: ${String(changeKind)}`);
  }
  const validFrom = parseInstant(mapping.validFrom);
  const validUntil = parseInstant(mapping.validUntil);
  if (mapping.validFrom !== null && validFrom === null) {
    throw new ModelMappingError("mapping validFrom must be a valid ISO timestamp");
  }
  if (mapping.validUntil !== null && validUntil === null) {
    throw new ModelMappingError("mapping validUntil must be a valid ISO timestamp");
  }
  if (validFrom !== null && validUntil !== null && validUntil <= validFrom) {
    throw new ModelMappingError("mapping validity interval must be half-open and non-empty");
  }
  if (changeKind === "historical_correction" && (validFrom === null || validUntil === null)) {
    throw new ModelMappingError(
      "historical correction mappings require both validFrom and validUntil",
    );
  }
  if (
    mapping.supersedesVersion !== null &&
    mapping.supersedesVersion !== undefined &&
    mapping.supersedesVersion === mapping.version
  ) {
    throw new ModelMappingError("mapping cannot supersede itself");
  }
  if (
    mapping.correctionOfVersion !== null &&
    mapping.correctionOfVersion !== undefined &&
    mapping.correctionOfVersion === mapping.version
  ) {
    throw new ModelMappingError("mapping cannot correct itself");
  }
  if (isMappingPublished(mapping)) {
    if (
      mapping.publishedAt === null ||
      mapping.publishedAt === undefined ||
      !isValidInstant(mapping.publishedAt)
    ) {
      throw new ModelMappingError(
        "published mappings require a valid publishedAt timestamp",
      );
    }
    if (
      mapping.reviewedAt === null ||
      mapping.reviewedAt === undefined ||
      !isValidInstant(mapping.reviewedAt) ||
      mapping.reviewedBy === null ||
      mapping.reviewedBy === undefined ||
      !mapping.reviewedBy.trim()
    ) {
      throw new ModelMappingError(
        "published mappings require reviewer identity and timestamp",
      );
    }
  } else if (mapping.publishedAt !== null && mapping.publishedAt !== undefined) {
    throw new ModelMappingError("draft mappings must not have publishedAt");
  }
}

function mappingApplicabilityWarnings(
  mapping: ModelMappingVersion,
  at: string | null,
): string[] {
  const hasValidityBound =
    (mapping.validFrom !== null && mapping.validFrom !== undefined) ||
    (mapping.validUntil !== null && mapping.validUntil !== undefined);
  if (mapping.reviewStatus === "retired" && at === null) {
    return ["retired_mapping_requires_event_time"];
  }
  if (hasValidityBound && at === null) {
    return ["mapping_requires_event_time"];
  }
  if (at !== null && !isValidInstant(at)) {
    throw new ModelMappingError("mapping application time must be a valid ISO timestamp");
  }
  if (at === null) {
    return [];
  }
  const instant = new Date(at).getTime();
  const validFrom =
    mapping.validFrom === null || mapping.validFrom === undefined
      ? null
      : new Date(mapping.validFrom).getTime();
  const validUntil =
    mapping.validUntil === null || mapping.validUntil === undefined
      ? null
      : new Date(mapping.validUntil).getTime();
  if (
    (validFrom !== null && instant < validFrom) ||
    (validUntil !== null && instant >= validUntil)
  ) {
    return ["mapping_outside_validity_interval"];
  }
  return [];
}

function mappingSelector(rule: MappingRule): string {
  return JSON.stringify([
    rule.collectorAccountId ?? null,
    rule.slug,
    rule.mode ?? null,
    rule.reasoningEffort ?? null,
  ]);
}

function matchesOptional(
  ruleValue: string | null | undefined,
  observed: string | null,
): boolean {
  return ruleValue === undefined || ruleValue === null || ruleValue === observed;
}

function rulesOverlap(left: MappingRule, right: MappingRule): boolean {
  return (
    left.slug === right.slug &&
    dimensionsOverlap(left.collectorAccountId, right.collectorAccountId) &&
    dimensionsOverlap(left.mode, right.mode) &&
    dimensionsOverlap(left.reasoningEffort, right.reasoningEffort)
  );
}

function dimensionsOverlap(
  left: string | null | undefined,
  right: string | null | undefined,
): boolean {
  return (
    left === undefined ||
    left === null ||
    right === undefined ||
    right === null ||
    left === right
  );
}

function compareRulePrecedence(left: MappingRule, right: MappingRule): number {
  const leftAccount = hasValue(left.collectorAccountId) ? 1 : 0;
  const rightAccount = hasValue(right.collectorAccountId) ? 1 : 0;
  if (leftAccount !== rightAccount) {
    return leftAccount - rightAccount;
  }
  const leftSpecificity = exactFieldCount(left);
  const rightSpecificity = exactFieldCount(right);
  if (leftSpecificity !== rightSpecificity) {
    return leftSpecificity - rightSpecificity;
  }
  return 0;
}

function exactFieldCount(rule: MappingRule): number {
  return Number(hasValue(rule.mode)) + Number(hasValue(rule.reasoningEffort));
}

function hasValue(value: string | null | undefined): value is string {
  return value !== undefined && value !== null;
}

function parseInstant(value: string | null | undefined): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  const parsed = new Date(value);
  return Number.isFinite(parsed.getTime()) ? parsed.getTime() : null;
}

function isValidInstant(value: string | null | undefined): boolean {
  return parseInstant(value) !== null;
}
