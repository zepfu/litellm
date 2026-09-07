import { describe, expect, it } from "vitest";

import type {
  MappingRule,
  ModelMappingVersion,
} from "../../src/ledger/types.js";
import {
  mapModelEvidence,
  resolveModelEvidence,
  validateMappingVersion,
} from "../../src/normalize/model-mapping.js";

describe("model mapping lifecycle", () => {
  it("fails closed for draft mappings and never applies their rules", () => {
    const mapping = mappingVersion({
      reviewStatus: "draft",
      rules: [reviewedRule("model-requested", "astra_pro")],
    });

    const result = resolveModelEvidence(
      { slug: "model-requested", mode: null, reasoningEffort: null },
      mapping,
      "account-one",
    );

    expect(result).toMatchObject({
      family: null,
      applied: false,
      warnings: ["mapping_unreviewed"],
    });
    expect(
      mapModelEvidence(
        { slug: "model-requested", mode: null, reasoningEffort: null },
        mapping,
        "account-one",
      ),
    ).toBeNull();
  });

  it("gives an explicit account override precedence over a generic rule", () => {
    const mapping = mappingVersion({
      rules: [
        reviewedRule("model-requested", "other_chat"),
        reviewedRule("model-requested", "astra_pro", {
          collectorAccountId: "account-one",
        }),
      ],
    });

    expect(
      mapModelEvidence(
        { slug: "model-requested", mode: null, reasoningEffort: null },
        mapping,
        "account-one",
      ),
    ).toBe("astra_pro");
    expect(
      mapModelEvidence(
        { slug: "model-requested", mode: null, reasoningEffort: null },
        mapping,
        "account-two",
      ),
    ).toBe("other_chat");
  });

  it("rejects overlapping rules with equal precedence and different families", () => {
    const mapping = mappingVersion({
      rules: [
        reviewedRule("model-requested", "astra_pro", { mode: "fast" }),
        reviewedRule("model-requested", "sol_pro", {
          reasoningEffort: "high",
        }),
      ],
    });

    expect(() => validateMappingVersion(mapping)).toThrow(
      /ambiguous mapping rule overlap/,
    );
  });

  it("rejects unreviewed rules in a published version", () => {
    const mapping = mappingVersion({
      rules: [reviewedRule("model-requested", "astra_pro", { reviewed: false })],
    });

    expect(() => validateMappingVersion(mapping)).toThrow(
      /published mapping rule is not reviewed/,
    );
  });

  it("requires an event time before applying a bounded prospective mapping", () => {
    const mapping = mappingVersion({
      validFrom: "2026-09-08T00:00:00.000Z",
      rules: [reviewedRule("model-requested", "astra_pro")],
    });

    expect(
      resolveModelEvidence(
        { slug: "model-requested", mode: null, reasoningEffort: null },
        mapping,
        "account-one",
      ),
    ).toMatchObject({
      applied: false,
      warnings: ["mapping_requires_event_time"],
    });
    expect(
      resolveModelEvidence(
        { slug: "model-requested", mode: null, reasoningEffort: null },
        mapping,
        "account-one",
        { at: "2026-09-07T23:59:59.999Z" },
      ),
    ).toMatchObject({
      applied: false,
      warnings: ["mapping_outside_validity_interval"],
    });
    expect(
      resolveModelEvidence(
        { slug: "model-requested", mode: null, reasoningEffort: null },
        mapping,
        "account-one",
        { at: "2026-09-08T00:00:00.000Z" },
      ),
    ).toMatchObject({ family: "astra_pro", applied: true });
  });

  it("requires a bounded interval for historical corrections", () => {
    const mapping = mappingVersion({
      changeKind: "historical_correction",
      rules: [reviewedRule("model-requested", "astra_pro")],
    });

    expect(() => validateMappingVersion(mapping)).toThrow(
      /historical correction mappings require both validFrom and validUntil/,
    );
  });
});

function mappingVersion(
  overrides: Partial<ModelMappingVersion> = {},
): ModelMappingVersion {
  return {
    version: "mapping-test",
    canonicalFamilies: ["astra_pro", "sol_pro", "other_chat", "unknown"],
    rules: [],
    reviewStatus: "approved",
    source: "unit-test",
    createdAt: "2026-09-07T00:00:00.000Z",
    reviewedAt: "2026-09-07T00:00:00.000Z",
    reviewedBy: "unit-test",
    ...overrides,
  };
}

function reviewedRule(
  slug: string,
  family: string,
  overrides: Partial<MappingRule> = {},
): MappingRule {
  return {
    slug,
    family,
    reviewed: true,
    source: "unit-test",
    ...overrides,
  };
}
