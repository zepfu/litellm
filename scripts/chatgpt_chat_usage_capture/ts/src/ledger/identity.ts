import { createHash } from "node:crypto";

import type { LedgerScope } from "./types.js";

export function scopeKey(scope: LedgerScope): string {
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

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
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
