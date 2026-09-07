/**
 * Shared contracts for the Stage-4 local API layer.
 *
 * The API is a thin security and routing shell over the existing collector
 * domain services. Handlers never touch the provider directly and never
 * fabricate payloads: routes whose backing service is not injected return an
 * explicit `unavailable` response. These types intentionally mirror the
 * build-spec 15.2 route list without adding provider egress of any kind.
 */

/** Explicit 503 response for a route with no injected service. */
export const UNAVAILABLE_REASON =
  "service_unavailable_no_injected_handler" as const;

/** Explicit 501 response for a route reserved by the spec but not yet owned. */
export const NOT_IMPLEMENTED_REASON = "not_implemented_reserved_route" as const;

/** Machine-readable API error codes. */
export type ApiErrorCode =
  | "invalid_request"
  | "unauthorized"
  | "forbidden"
  | "not_found"
  | "conflict"
  | "unavailable"
  | "not_implemented";

/** Structured JSON error envelope; never contains secrets or stack traces. */
export interface ApiErrorBody {
  error: {
    code: ApiErrorCode;
    message: string;
  };
}

/** HTTP status codes used by the local API. */
export const HTTP_STATUS = {
  ok: 200,
  accepted: 202,
  badRequest: 400,
  unauthorized: 401,
  forbidden: 403,
  notFound: 404,
  methodNotAllowed: 405,
  conflict: 409,
  unavailable: 503,
  notImplemented: 501,
} as const;

/** Validated query for GET /api/v1/usage. */
export interface UsageQuery {
  account?: string | undefined;
  from?: string | undefined;
  to?: string | undefined;
  groupBy?: string | undefined;
}

/** Validated query for GET /api/v1/attempts. */
export interface AttemptsQuery {
  account?: string | undefined;
  cursor?: string | undefined;
  limit?: number | undefined;
}

/** Validated query for GET /api/v1/coverage. */
export interface CoverageQuery {
  account?: string | undefined;
  from?: string | undefined;
  to?: string | undefined;
}

/** A single read or write operation exposed by the local API. */
export interface LocalApiServices {
  /** GET /api/v1/accounts */
  listAccounts?(): Promise<unknown>;
  /** GET /api/v1/status?account=... */
  getStatus?(query: { account?: string | undefined }): Promise<unknown>;
  /** GET /api/v1/usage?account=...&from=...&to=...&group_by=... */
  getUsage?(query: UsageQuery): Promise<unknown>;
  /** GET /api/v1/attempts?account=...&cursor=...&limit=... */
  getAttempts?(query: AttemptsQuery): Promise<unknown>;
  /** GET /api/v1/quota-windows?account=... */
  getQuotaWindows?(query: { account?: string | undefined }): Promise<unknown>;
  /** GET /api/v1/quota-observations?account=... */
  getQuotaObservations?(query: {
    account?: string | undefined;
  }): Promise<unknown>;
  /** GET /api/v1/coverage?account=...&from=...&to=... */
  getCoverage?(query: CoverageQuery): Promise<unknown>;
  /** GET /api/v1/runs?account=... */
  getRuns?(query: { account?: string | undefined }): Promise<unknown>;

  /** POST /api/v1/refresh — request a collection refresh for an account. */
  requestRefresh?(input: {
    account?: string | undefined;
    idempotencyKey?: string | undefined;
  }): Promise<unknown>;
  /** POST /api/v1/schedule — replace the refresh cadence. Owned by scheduler. */
  updateSchedule?(input: {
    account?: string | undefined;
    intervalHours?: number | undefined;
    idempotencyKey?: string | undefined;
  }): Promise<unknown>;
  /** POST /api/v1/window-definitions — record reset-window evidence. */
  upsertWindowDefinition?(input: {
    account?: string | undefined;
    window?: Record<string, unknown> | undefined;
    idempotencyKey?: string | undefined;
  }): Promise<unknown>;
  /** POST /api/v1/manual-observations — record an operator quota observation. */
  recordManualObservation?(input: {
    account?: string | undefined;
    observation?: Record<string, unknown> | undefined;
    idempotencyKey?: string | undefined;
  }): Promise<unknown>;
  /** POST /api/v1/rebuild-preview — dry-run accounting rebuild. */
  rebuildPreview?(input: {
    account?: string | undefined;
    idempotencyKey?: string | undefined;
  }): Promise<unknown>;
  /** POST /api/v1/rebuild-apply — commit an accounting rebuild revision. */
  rebuildApply?(input: {
    account?: string | undefined;
    previewId?: string | undefined;
    idempotencyKey?: string | undefined;
  }): Promise<unknown>;

  /** Liveness: process is running. Returns quickly. */
  checkLive?(): Promise<unknown>;
  /** Readiness: database/config usable. */
  checkReady?(): Promise<unknown>;
}

/** Configuration for the local API server. */
export interface LocalApiServerOptions {
  /** Loopback-only token required for every /api/v1/* route. */
  authToken: string;
  /** Bind address. Defaults to 127.0.0.1 (loopback only). */
  host?: string | undefined;
  /** Port. Defaults to 0 (ephemeral). */
  port?: number | undefined;
  /**
   * Explicit opt-in to bind a non-loopback interface. Required when `host`
   * is not loopback; otherwise the server refuses to start.
   */
  allowRemoteBind?: boolean | undefined;
  /**
   * Allowed Host header values (exact match, case-insensitive). When unset,
   * only loopback hostnames and the bound host are accepted.
   */
  allowedHosts?: readonly string[] | undefined;
  /**
   * Allowed Origin header values for browser clients. When unset, only
   * loopback origins are accepted.
   */
  allowedOrigins?: readonly string[] | undefined;
  /** Injected domain services. Missing handlers return explicit 503. */
  services: LocalApiServices;
  /** Maximum JSON body bytes for POST routes. Default 16 KiB. */
  maxBodyBytes?: number | undefined;
  /** Maximum query-string length. Default 2048. */
  maxQueryLength?: number | undefined;
  /** Maximum pagination limit for list routes. Default 200, hard cap 1000. */
  maxPageLimit?: number | undefined;
  /** Maximum idempotency-cache entries. Default 1000. */
  maxIdempotencyEntries?: number | undefined;
  /** Maximum idempotency-key length. Default 128. */
  maxIdempotencyKeyLength?: number | undefined;
}

/** Handle for a running local API server. */
export interface LocalApiServerHandle {
  /** Bound address information. */
  address(): { host: string; port: number };
  /** Stop accepting connections and close idle ones. */
  close(): Promise<void>;
}

/** Loopback hostnames/addresses accepted by default. */
export const LOOPBACK_HOSTS = new Set([
  "localhost",
  "127.0.0.1",
  "[::1]",
  "::1",
]);

/** Default pagination cap for list routes. */
export const DEFAULT_MAX_PAGE_LIMIT = 200;
/** Hard ceiling for pagination regardless of configuration. */
export const HARD_MAX_PAGE_LIMIT = 1000;
/** Default JSON body cap for POST routes. */
export const DEFAULT_MAX_BODY_BYTES = 16 * 1024;
/** Default query-string cap. */
export const DEFAULT_MAX_QUERY_LENGTH = 2048;
