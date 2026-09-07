/**
 * Public exports for the local API layer.
 */

export {
  createLocalApiServer,
} from "./server.js";

export {
  IdempotencyCache,
  fingerprintRequest,
} from "./idempotency.js";

export {
  isLoopbackHost,
  isAuthenticated,
  isHostAllowed,
  isOriginAllowed,
  hasCsrfHeader,
  parseHostHeader,
} from "./security.js";

export {
  ApiValidationError,
  readJsonBody,
  parseQuery,
  optionalAccountId,
  optionalCursor,
  optionalIdempotencyKey,
  optionalLimit,
  optionalQueryParam,
  optionalTimestamp,
} from "./validation.js";

export type {
  ApiErrorBody,
  ApiErrorCode,
  AttemptsQuery,
  CoverageQuery,
  LocalApiServerHandle,
  LocalApiServerOptions,
  LocalApiServices,
  UsageQuery,
} from "./types.js";

export {
  HTTP_STATUS,
  LOOPBACK_HOSTS,
  UNAVAILABLE_REASON,
  NOT_IMPLEMENTED_REASON,
  DEFAULT_MAX_BODY_BYTES,
  DEFAULT_MAX_PAGE_LIMIT,
  DEFAULT_MAX_QUERY_LENGTH,
  HARD_MAX_PAGE_LIMIT,
} from "./types.js";
