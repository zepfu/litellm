/**
 * Bounded, dependency-free validation for the local API.
 *
 * All query values are strings, bodies are plain JSON objects with bounded
 * size and nesting depth, and timestamps must be valid RFC 3339 strings.
 * Everything crossing the HTTP boundary is untrusted input.
 */

import type { IncomingMessage } from "node:http";

import { HTTP_STATUS, type ApiErrorBody } from "./types.js";

/** Thrown by validators; converted into a structured JSON error. */
export class ApiValidationError extends Error {
  readonly status: number;
  readonly code: ApiErrorBody["error"]["code"];

  constructor(
    status: number,
    code: ApiErrorBody["error"]["code"],
    message: string,
  ) {
    super(message);
    this.name = "ApiValidationError";
    this.status = status;
    this.code = code;
  }
}

/** Maximum accepted JSON nesting depth. */
export const MAX_JSON_DEPTH = 16;
/** Maximum accepted number of keys in a JSON object or array. */
export const MAX_JSON_KEYS = 256;
/** Maximum accepted query-string length (bounded before parsing). */
export const MAX_QUERY_STRING_LENGTH = 2048;
/** Maximum accepted cursor length. */
export const MAX_CURSOR_LENGTH = 512;
/** Maximum accepted account-id length. */
export const MAX_ACCOUNT_ID_LENGTH = 128;

const RFC3339 =
  /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/;

/** Validate a single query parameter string. */
export function requireQueryString(
  value: string | undefined,
  name: string,
): string {
  if (value === undefined) {
    throw new ApiValidationError(
      HTTP_STATUS.badRequest,
      "invalid_request",
      `missing required query parameter: ${name}`,
    );
  }
  return value;
}

/** Validate an optional account identifier. */
export function optionalAccountId(value: string | undefined): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (value.length === 0 || value.length > MAX_ACCOUNT_ID_LENGTH) {
    throw new ApiValidationError(
      HTTP_STATUS.badRequest,
      "invalid_request",
      "account must be 1..128 characters",
    );
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value)) {
    throw new ApiValidationError(
      HTTP_STATUS.badRequest,
      "invalid_request",
      "account contains invalid characters",
    );
  }
  return value;
}

/** Validate an optional RFC 3339 timestamp. */
export function optionalTimestamp(
  value: string | undefined,
  name: string,
): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!RFC3339.test(value) || Number.isNaN(Date.parse(value))) {
    throw new ApiValidationError(
      HTTP_STATUS.badRequest,
      "invalid_request",
      `${name} must be a valid RFC 3339 timestamp`,
    );
  }
  return value;
}

/** Validate an optional pagination cursor. */
export function optionalCursor(value: string | undefined): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (value.length === 0 || value.length > MAX_CURSOR_LENGTH) {
    throw new ApiValidationError(
      HTTP_STATUS.badRequest,
      "invalid_request",
      `cursor must be 1..${MAX_CURSOR_LENGTH} characters`,
    );
  }
  return value;
}

/** Validate an optional positive-integer limit, clamped to a hard cap. */
export function optionalLimit(
  value: string | undefined,
  defaultLimit: number,
  hardCap: number,
): number {
  if (value === undefined) {
    return Math.min(defaultLimit, hardCap);
  }
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > hardCap) {
    throw new ApiValidationError(
      HTTP_STATUS.badRequest,
      "invalid_request",
      `limit must be an integer in 1..${hardCap}`,
    );
  }
  return parsed;
}

/** Validate an optional idempotency key. */
export function optionalIdempotencyKey(
  value: string | undefined,
  maxLength: number,
): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (value.length === 0 || value.length > maxLength) {
    throw new ApiValidationError(
      HTTP_STATUS.badRequest,
      "invalid_request",
      `Idempotency-Key must be 1..${maxLength} characters`,
    );
  }
  return value;
}

/** Read a bounded JSON request body. */
export async function readJsonBody(
  request: IncomingMessage,
  maxBodyBytes: number,
): Promise<Record<string, unknown>> {
  const contentType = request.headers["content-type"];
  const normalized = Array.isArray(contentType)
    ? contentType[0]
    : contentType;
  if (
    normalized !== undefined &&
    !normalized.toLowerCase().startsWith("application/json")
  ) {
    throw new ApiValidationError(
      HTTP_STATUS.badRequest,
      "invalid_request",
      "Content-Type must be application/json",
    );
  }

  const body = await new Promise<Buffer>((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    request.on("data", (chunk: Buffer) => {
      total += chunk.length;
      if (total > maxBodyBytes) {
        reject(
          new ApiValidationError(
            HTTP_STATUS.badRequest,
            "invalid_request",
            `request body exceeds ${maxBodyBytes} bytes`,
          ),
        );
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => resolve(Buffer.concat(chunks)));
    request.on("error", (err: Error) => reject(err));
  });

  if (body.length === 0) {
    return {};
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(body.toString("utf8"));
  } catch {
    throw new ApiValidationError(
      HTTP_STATUS.badRequest,
      "invalid_request",
      "request body must be valid JSON",
    );
  }

  return assertPlainObject(parsed, 0);
}

/** Recursively assert a plain JSON object with bounded depth/keys. */
function assertPlainObject(value: unknown, depth: number): Record<string, unknown> {
  if (depth > MAX_JSON_DEPTH) {
    throw new ApiValidationError(
      HTTP_STATUS.badRequest,
      "invalid_request",
      `JSON nesting exceeds ${MAX_JSON_DEPTH}`,
    );
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ApiValidationError(
      HTTP_STATUS.badRequest,
      "invalid_request",
      "request body must be a JSON object",
    );
  }
  const keys = Object.keys(value);
  if (keys.length > MAX_JSON_KEYS) {
    throw new ApiValidationError(
      HTTP_STATUS.badRequest,
      "invalid_request",
      `JSON object has more than ${MAX_JSON_KEYS} keys`,
    );
  }
  for (const key of keys) {
    const child = (value as Record<string, unknown>)[key];
    if (typeof child === "object" && child !== null) {
      if (Array.isArray(child)) {
        if (child.length > MAX_JSON_KEYS) {
          throw new ApiValidationError(
            HTTP_STATUS.badRequest,
            "invalid_request",
            `JSON array has more than ${MAX_JSON_KEYS} items`,
          );
        }
        for (const item of child) {
          if (typeof item === "object" && item !== null) {
            assertPlainObject(item, depth + 1);
          }
        }
      } else {
        assertPlainObject(child, depth + 1);
      }
    }
  }
  return value as Record<string, unknown>;
}

/** Parse and validate the query string of a request URL. */
export function parseQuery(
  rawQuery: string,
  maxLength: number,
): URLSearchParams {
  if (rawQuery.length > maxLength) {
    throw new ApiValidationError(
      HTTP_STATUS.badRequest,
      "invalid_request",
      `query string exceeds ${maxLength} characters`,
    );
  }
  return new URLSearchParams(rawQuery);
}

/** Extract a single optional string parameter. */
export function optionalQueryParam(
  params: URLSearchParams,
  name: string,
): string | undefined {
  const value = params.get(name);
  return value === null ? undefined : value;
}
