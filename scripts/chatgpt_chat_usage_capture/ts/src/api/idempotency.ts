/**
 * In-memory idempotency cache for local POST routes.
 *
 * Replays with the same key and request body return the original response.
 * Replays with the same key but a different body are rejected as conflicts.
 * Entries are bounded and evicted oldest-first.
 */

import { createHash } from "node:crypto";

export interface IdempotencyEntry {
  readonly fingerprint: string;
  readonly status: number;
  readonly body: unknown;
}

interface CompleteEntry extends IdempotencyEntry {
  readonly state: "complete";
}

interface PendingEntry {
  readonly state: "pending";
  readonly fingerprint: string;
  readonly promise: Promise<IdempotencyEntry>;
  readonly resolve: (entry: IdempotencyEntry) => void;
  readonly reject: (reason?: unknown) => void;
}

type CacheEntry = CompleteEntry | PendingEntry;

export type IdempotencyReservation =
  | {
      readonly kind: "new";
      readonly complete: (status: number, body: unknown) => IdempotencyEntry;
      readonly fail: (reason: unknown) => void;
    }
  | {
      readonly kind: "pending";
      readonly promise: Promise<IdempotencyEntry>;
    }
  | {
      readonly kind: "replay";
      readonly entry: IdempotencyEntry;
    }
  | {
      readonly kind: "conflict";
    }
  | {
      readonly kind: "capacity";
    };

/** Bounded idempotency cache keyed by route + key. */
export class IdempotencyCache {
  private readonly entries = new Map<string, CacheEntry>();
  private readonly maxEntries: number;

  constructor(maxEntries: number) {
    this.maxEntries = Math.max(1, Math.floor(maxEntries));
  }

  /**
   * Look up a completed response. Pending reservations are intentionally not
   * exposed through this compatibility helper; use reserve() for writes.
   */
  get(route: string, key: string): IdempotencyEntry | undefined {
    const entry = this.entries.get(cacheKey(route, key));
    return entry?.state === "complete" ? exposeEntry(entry) : undefined;
  }

  /**
   * Reserve a key before invoking the callback.
   *
   * The synchronous map insertion is the single-flight boundary: another
   * request cannot claim the same key until this request completes or fails.
   */
  reserve(
    route: string,
    key: string,
    requestFingerprint: string,
  ): IdempotencyReservation {
    const keyName = cacheKey(route, key);
    const existing = this.entries.get(keyName);
    if (existing !== undefined) {
      if (existing.fingerprint !== requestFingerprint) {
        return { kind: "conflict" };
      }
      if (existing.state === "pending") {
        return { kind: "pending", promise: existing.promise };
      }
      return { kind: "replay", entry: exposeEntry(existing) };
    }

    if (this.entries.size >= this.maxEntries && !this.evictOldestComplete()) {
      return { kind: "capacity" };
    }

    let resolvePending!: (entry: IdempotencyEntry) => void;
    let rejectPending!: (reason?: unknown) => void;
    const promise = new Promise<IdempotencyEntry>((resolve, reject) => {
      resolvePending = resolve;
      rejectPending = reject;
    });
    // The owner also handles the rejection. This prevents an unused pending
    // promise from becoming an unhandled rejection when no waiter exists.
    void promise.catch(() => undefined);

    const pending: PendingEntry = {
      state: "pending",
      fingerprint: requestFingerprint,
      promise,
      resolve: resolvePending,
      reject: rejectPending,
    };
    this.entries.set(keyName, pending);

    let settled = false;
    return {
      kind: "new",
      complete: (status, body) => {
        if (settled) {
          throw new Error("Idempotency reservation already settled");
        }
        if (this.entries.get(keyName) !== pending) {
          throw new Error("Idempotency reservation is no longer active");
        }
        const complete = makeCompleteEntry(
          requestFingerprint,
          status,
          body,
        );
        settled = true;
        this.entries.set(keyName, complete);
        const exposed = exposeEntry(complete);
        pending.resolve(exposed);
        return exposed;
      },
      fail: (reason) => {
        if (settled) {
          return;
        }
        settled = true;
        if (this.entries.get(keyName) === pending) {
          this.entries.delete(keyName);
        }
        pending.reject(reason);
      },
    };
  }

  /**
   * Store a completed response for compatibility with callers that do not
   * need single-flight behavior. New write paths should use reserve().
   */
  set(
    route: string,
    key: string,
    status: number,
    body: unknown,
    requestFingerprint: string,
  ): void {
    const keyName = cacheKey(route, key);
    const existing = this.entries.get(keyName);
    if (existing?.state === "pending") {
      throw new Error("Cannot overwrite an in-flight idempotency reservation");
    }
    if (existing === undefined &&
        this.entries.size >= this.maxEntries &&
        !this.evictOldestComplete()) {
      throw new Error("Idempotency cache is full");
    }
    this.entries.set(
      keyName,
      makeCompleteEntry(requestFingerprint, status, body),
    );
  }

  private evictOldestComplete(): boolean {
    for (const [key, entry] of this.entries) {
      if (entry.state === "complete") {
        this.entries.delete(key);
        return true;
      }
    }
    return false;
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

/** Clone a value before handing it to a callback or another request. */
export function cloneValue<T>(value: T): T {
  return structuredClone(value);
}

/** Clone and recursively freeze a deterministic request/response snapshot. */
export function immutableSnapshot<T>(value: T): T {
  return deepFreeze(cloneValue(value));
}

function stableSerialize(value: unknown): string {
  if (value === null) {
    return "null";
  }
  if (typeof value !== "object") {
    if (typeof value === "string") {
      return JSON.stringify(value);
    }
    if (typeof value === "undefined") {
      return "undefined";
    }
    if (typeof value === "number") {
      return JSON.stringify(value) ?? "null";
    }
    if (typeof value === "bigint") {
      return `${value}n`;
    }
    return `${typeof value}:${String(value)}`;
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

function cacheKey(route: string, key: string): string {
  return `${route}\n${key}`;
}

function makeCompleteEntry(
  fingerprint: string,
  status: number,
  body: unknown,
): CompleteEntry {
  return Object.freeze({
    state: "complete" as const,
    fingerprint,
    status,
    body: immutableSnapshot(body),
  });
}

function exposeEntry(entry: CompleteEntry): IdempotencyEntry {
  return Object.freeze({
    fingerprint: entry.fingerprint,
    status: entry.status,
    body: immutableSnapshot(entry.body),
  });
}

function deepFreeze<T>(value: T): T {
  if (
    value === null ||
    (typeof value !== "object" && typeof value !== "function") ||
    Object.isFrozen(value)
  ) {
    return value;
  }

  Object.freeze(value);
  if (Array.isArray(value)) {
    for (const item of value) {
      deepFreeze(item);
    }
  } else {
    for (const child of Object.values(value as Record<string, unknown>)) {
      deepFreeze(child);
    }
  }
  return value;
}
