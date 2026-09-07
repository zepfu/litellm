import { createHash } from "node:crypto";

import type { LedgerScope } from "./types.js";

export function scopeKey(scope: LedgerScope): string {
  return activityKey(scope);
}

/**
 * Stable identity for provider activity belonging to one verified owner.
 *
 * The local collector account is deliberately excluded so two independently
 * configured collectors for the same verified owner converge on one activity
 * ledger. Unverified scopes remain collector-local rather than being merged.
 */
export function activityKey(scope: LedgerScope): string {
  const owner = [
    scope.provider,
    scope.providerUserId,
    scope.workspaceId,
    scope.quotaOwnerId,
    scope.surface,
  ];
  if (
    scope.providerUserId === null ||
    scope.workspaceId === null ||
    scope.quotaOwnerId === null
  ) {
    owner.unshift("unverified", scope.collectorAccountId);
  }
  return sha256(
    JSON.stringify(owner),
  );
}

/** Identity used only where local collector rows must remain distinct. */
export function collectorScopeKey(scope: LedgerScope): string {
  return sha256(
    JSON.stringify([
      scope.collectorAccountId,
      scope.provider,
      scope.providerUserId,
      scope.workspaceId,
      scope.quotaOwnerId,
      scope.surface,
    ]),
  );
}

export function stableId(...parts: string[]): string {
  return sha256(JSON.stringify(parts));
}

export function canonicalJson(value: unknown): string {
  return JSON.stringify(sortValue(value));
}

export function fingerprint(value: unknown): string {
  return sha256(canonicalJson(value));
}

const PROVENANCE_ALLOWLIST = new Set([
  "adapter_version",
  "collector",
  "collector_account_id",
  "coverage",
  "detail_route",
  "endpoint",
  "fixture",
  "page_kind",
  "pagination_state",
  "revisit_reason",
  "route",
  "schema_version",
  "scopes",
  "source",
  "source_id",
  "source_kind",
  "surface",
]);

export type SanitizedProvenance = Record<
  string,
  boolean | number | string | string[]
>;

/**
 * Provenance is diagnostic metadata, not an open-ended payload channel.
 * Unknown keys and unsafe values are dropped before JSON persistence.
 */
export function sanitizeProvenance(
  value: Record<string, unknown> | undefined,
): SanitizedProvenance {
  const out: SanitizedProvenance = {};
  if (!value) {
    return out;
  }
  for (const [key, rawValue] of Object.entries(value)) {
    if (!PROVENANCE_ALLOWLIST.has(key)) {
      continue;
    }
    const sanitized = sanitizeProvenanceValue(rawValue);
    if (sanitized !== undefined) {
      out[key] = sanitized;
    }
  }
  return out;
}

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function sanitizeProvenanceValue(
  value: unknown,
): boolean | number | string | string[] | undefined {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : undefined;
  }
  if (typeof value === "string") {
    const normalized = value.trim();
    return /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$/.test(normalized)
      ? normalized
      : undefined;
  }
  if (Array.isArray(value)) {
    const items = value
      .filter((item): item is string => typeof item === "string")
      .map((item) => item.trim())
      .filter((item) => /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$/.test(item))
      .slice(0, 32);
    return items;
  }
  return undefined;
}

function sortValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => sortValue(item));
  }
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return Object.fromEntries(
      Object.keys(record)
        .sort()
        .map((key) => [key, sortValue(record[key])]),
    );
  }
  return value;
}
