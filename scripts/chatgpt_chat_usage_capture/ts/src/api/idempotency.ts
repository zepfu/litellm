/**
 * In-memory idempotency cache for local POST routes.
 *
 * Replays with the same key and request body return the original response.
 * Replays with the same key but a different body are rejected as conflicts.
 * Entries are bounded and evicted oldest-first.
 */

import { createHash } from "node:crypto";

interface IdempotencyEntry {
  fingerprint: string;
  status: number;
  body: unknown;
}

/** Bounded idempotency cache keyed by route + key. */
export class IdempotencyCache {
  private readonly entries = new Map<string, IdempotencyEntry>();
  private readonly maxEntries: number;

  constructor(maxEntries: number) {
    this.maxEntries = Math.max(1, maxEntries);
  }

  /** Look up a cached response. */
  get(route: string, key: string): IdempotencyEntry | undefined {
    return this.entries.get(`${route}\n${key}`);
  }

  /** Store a response for a route + key. */
  set(
    route: string,
    key: string,
    status: number,
    body: unknown,
    requestFingerprint: string,
  ): void {
    const cacheKey = `${route}\n${key}`;
    if (this.entries.size >= this.maxEntries && !this.entries.has(cacheKey)) {
      const oldest = this.entries.keys().next();
      if (!oldest.done) {
        this.entries.delete(oldest.value);
      }
    }
    this.entries.set(cacheKey, {
      fingerprint: requestFingerprint,
      status,
      body,
    });
  }
}

/** Compute a stable fingerprint of the request body for conflict detection. */
export function fingerprintRequest(route: string, body: unknown): string {
  const hash = createHash("sha256");
  hash.update(route);
  hash.update("\n");
  hash.update(stableSerialize(body));
  return hash.digest("hex");
}

function stableSerialize(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableSerialize).join(",")}]`;
  }
  const record = value as Record<string, unknown>;
  const keys = Object.keys(record).sort();
  return `{${keys
    .map((key) => `${JSON.stringify(key)}:${stableSerialize(record[key])}`)
    .join(",")}}`;
}
