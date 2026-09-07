/**
 * Local HTTP API server for the ChatGPT Chat usage collector.
 *
 * This is a thin security and routing layer over the existing collector
 * domain services. It provides no provider egress, no arbitrary fetch
 * endpoint, and no HTML rendering. All routes are authenticated, loopback
 * by default, with explicit opt-in for remote binding, Host/Origin/CSRF
 * validation, bounded JSON/query inputs, and idempotent local writes.
 */

import { createServer, type Server } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";

import {
  cloneValue,
  fingerprintRequest,
  IdempotencyCache,
  immutableSnapshot,
} from "./idempotency.js";
import {
  hasCsrfHeader,
  isAuthenticated,
  isHostAllowed,
  isLoopbackHost,
  isOriginAllowed,
  parseHostHeader,
} from "./security.js";
import {
  ApiValidationError,
  optionalAccountId,
  optionalCursor,
  optionalIdempotencyKey,
  optionalLimit,
  optionalQueryParam,
  optionalTimestamp,
  parseQuery,
  readJsonBody,
} from "./validation.js";
import {
  DEFAULT_MAX_BODY_BYTES,
  DEFAULT_MAX_PAGE_LIMIT,
  DEFAULT_MAX_QUERY_LENGTH,
  HARD_MAX_PAGE_LIMIT,
  HTTP_STATUS,
  NOT_IMPLEMENTED_REASON,
  UNAVAILABLE_REASON,
  type ApiErrorBody,
  type LocalApiServerHandle,
  type LocalApiServerOptions,
  type LocalApiServices,
} from "./types.js";

const API_PREFIX = "/api/v1";
const JSON_HEADERS = {
  "Content-Type": "application/json; charset=utf-8",
  "Cache-Control": "no-store",
  "X-Content-Type-Options": "nosniff",
} as const;

/** Send a JSON response without exposing internals. */
function sendJson(
  response: ServerResponse,
  status: number,
  body: unknown,
): void {
  response.writeHead(status, JSON_HEADERS);
  response.end(JSON.stringify(body));
}

/** Send a structured error response. */
function sendError(
  response: ServerResponse,
  status: number,
  code: ApiErrorBody["error"]["code"],
  message: string,
): void {
  const body: ApiErrorBody = { error: { code, message } };
  sendJson(response, status, body);
}

/** Routes listed in build-spec 15.2 but owned by other stages. */
const RESERVED_ROUTES = new Set([
  `${API_PREFIX}/schedule`,
]);

/** Create and start the local API server. */
export async function createLocalApiServer(
  options: LocalApiServerOptions,
): Promise<LocalApiServerHandle> {
  const host = options.host ?? "127.0.0.1";
  const port = options.port ?? 0;
  const allowRemoteBind = options.allowRemoteBind ?? false;
  const maxBodyBytes = options.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES;
  const maxQueryLength = options.maxQueryLength ?? DEFAULT_MAX_QUERY_LENGTH;
  const maxPageLimit = Math.min(
    options.maxPageLimit ?? DEFAULT_MAX_PAGE_LIMIT,
    HARD_MAX_PAGE_LIMIT,
  );
  const maxIdempotencyEntries = options.maxIdempotencyEntries ?? 1000;
  const maxIdempotencyKeyLength = options.maxIdempotencyKeyLength ?? 128;
  const services = options.services;
  const idempotency = new IdempotencyCache(maxIdempotencyEntries);

  if (!isLoopbackHost(host) && !allowRemoteBind) {
    throw new Error(
      "Refusing to bind non-loopback host without explicit allowRemoteBind: true",
    );
  }

  const server = createServer(async (request, response) => {
    try {
      await routeRequest(request, response);
    } catch (error) {
      if (error instanceof ApiValidationError) {
        sendError(response, error.status, error.code, error.message);
      } else {
        sendError(
          response,
          HTTP_STATUS.unavailable,
          "unavailable",
          "internal error",
        );
      }
    }
  });

  await new Promise<void>((resolve, reject) => {
    server.listen(port, host, () => resolve());
    server.on("error", (error) => reject(error));
  });

  const addressInfo = server.address();
  if (addressInfo === null || typeof addressInfo === "string") {
    throw new Error("Server address is unavailable after listen");
  }
  const boundHost = addressInfo.address;
  const boundPort = addressInfo.port;

  return {
    address: () => ({ host: boundHost, port: boundPort }),
    close: () => closeServer(server),
  };

  async function routeRequest(
    request: IncomingMessage,
    response: ServerResponse,
  ): Promise<void> {
    const method = request.method ?? "GET";
    const rawUrl = request.url ?? "/";

    // Validate Host header for all requests.
    if (!isHostAllowed(request.headers.host, boundHost, options.allowedHosts)) {
      sendError(
        response,
        HTTP_STATUS.forbidden,
        "forbidden",
        "Host header is not allowed",
      );
      return;
    }

    // Validate Origin for browser requests.
    if (!isOriginAllowed(
      request.headers.origin,
      boundHost,
      boundPort,
      options.allowedOrigins,
    )) {
      sendError(
        response,
        HTTP_STATUS.forbidden,
        "forbidden",
        "Origin is not allowed",
      );
      return;
    }

    // Health endpoints are intentionally minimal and unauthenticated.
    if (rawUrl === "/health/live") {
      if (method !== "GET") {
        sendError(
          response,
          HTTP_STATUS.methodNotAllowed,
          "invalid_request",
          "Method not allowed",
        );
        return;
      }
      await handleHealth(response, services.checkLive, { status: "live" });
      return;
    }
    if (rawUrl === "/health/ready") {
      if (method !== "GET") {
        sendError(
          response,
          HTTP_STATUS.methodNotAllowed,
          "invalid_request",
          "Method not allowed",
        );
        return;
      }
      await handleHealth(response, services.checkReady, { status: "ready" });
      return;
    }

    // All API routes require authentication.
    if (!rawUrl.startsWith(API_PREFIX)) {
      sendError(response, HTTP_STATUS.notFound, "not_found", "Not found");
      return;
    }
    if (!isAuthenticated(request, options.authToken)) {
      sendError(
        response,
        HTTP_STATUS.unauthorized,
        "unauthorized",
        "Authentication required",
      );
      return;
    }

    // Mutating requests additionally require the CSRF header.
    if (method === "POST" && !hasCsrfHeader(request, options.authToken)) {
      sendError(
        response,
        HTTP_STATUS.forbidden,
        "forbidden",
        "CSRF header missing or invalid",
      );
      return;
    }

    const pathEnd = rawUrl.indexOf("?");
    const pathname = pathEnd >= 0 ? rawUrl.slice(0, pathEnd) : rawUrl;
    const rawQuery = pathEnd >= 0 ? rawUrl.slice(pathEnd + 1) : "";
    const params = parseQuery(rawQuery, maxQueryLength);

    if (RESERVED_ROUTES.has(pathname)) {
      sendError(
        response,
        HTTP_STATUS.notImplemented,
        "not_implemented",
        NOT_IMPLEMENTED_REASON,
      );
      return;
    }

    switch (pathname) {
      case `${API_PREFIX}/accounts`:
        await requireMethod(method, "GET", response, async () => {
          await requireService(services.listAccounts, response, async (service) => {
            sendJson(response, HTTP_STATUS.ok, await service());
          });
        });
        return;

      case `${API_PREFIX}/status`:
        await requireMethod(method, "GET", response, async () => {
          await requireService(services.getStatus, response, async (service) => {
            const account = optionalAccountId(
              optionalQueryParam(params, "account"),
            );
            sendJson(response, HTTP_STATUS.ok, await service({ account }));
          });
        });
        return;

      case `${API_PREFIX}/usage`:
        await requireMethod(method, "GET", response, async () => {
          await requireService(services.getUsage, response, async (service) => {
            const account = optionalAccountId(
              optionalQueryParam(params, "account"),
            );
            const from = optionalTimestamp(
              optionalQueryParam(params, "from"),
              "from",
            );
            const to = optionalTimestamp(
              optionalQueryParam(params, "to"),
              "to",
            );
            const groupBy = optionalQueryParam(params, "group_by");
            sendJson(
              response,
              HTTP_STATUS.ok,
              await service({ account, from, to, groupBy }),
            );
          });
        });
        return;

      case `${API_PREFIX}/attempts`:
        await requireMethod(method, "GET", response, async () => {
          await requireService(services.getAttempts, response, async (service) => {
            const account = optionalAccountId(
              optionalQueryParam(params, "account"),
            );
            const cursor = optionalCursor(
              optionalQueryParam(params, "cursor"),
            );
            const limit = optionalLimit(
              optionalQueryParam(params, "limit"),
              maxPageLimit,
              HARD_MAX_PAGE_LIMIT,
            );
            sendJson(
              response,
              HTTP_STATUS.ok,
              await service({ account, cursor, limit }),
            );
          });
        });
        return;

      case `${API_PREFIX}/quota-windows`:
        await requireMethod(method, "GET", response, async () => {
          await requireService(services.getQuotaWindows, response, async (service) => {
            const account = optionalAccountId(
              optionalQueryParam(params, "account"),
            );
            sendJson(response, HTTP_STATUS.ok, await service({ account }));
          });
        });
        return;

      case `${API_PREFIX}/quota-observations`:
        await requireMethod(method, "GET", response, async () => {
          await requireService(services.getQuotaObservations, response, async (service) => {
            const account = optionalAccountId(
              optionalQueryParam(params, "account"),
            );
            sendJson(response, HTTP_STATUS.ok, await service({ account }));
          });
        });
        return;

      case `${API_PREFIX}/coverage`:
        await requireMethod(method, "GET", response, async () => {
          await requireService(services.getCoverage, response, async (service) => {
            const account = optionalAccountId(
              optionalQueryParam(params, "account"),
            );
            const from = optionalTimestamp(
              optionalQueryParam(params, "from"),
              "from",
            );
            const to = optionalTimestamp(
              optionalQueryParam(params, "to"),
              "to",
            );
            sendJson(
              response,
              HTTP_STATUS.ok,
              await service({ account, from, to }),
            );
          });
        });
        return;

      case `${API_PREFIX}/runs`:
        await requireMethod(method, "GET", response, async () => {
          await requireService(services.getRuns, response, async (service) => {
            const account = optionalAccountId(
              optionalQueryParam(params, "account"),
            );
            sendJson(response, HTTP_STATUS.ok, await service({ account }));
          });
        });
        return;

      case `${API_PREFIX}/refresh`:
        await requireMethod(method, "POST", response, async () => {
          await requireService(services.requestRefresh, response, async (service) => {
            await handleIdempotentPost(
              request,
              response,
              service,
              idempotency,
              "refresh",
              maxBodyBytes,
              maxIdempotencyKeyLength,
            );
          });
        });
        return;

      case `${API_PREFIX}/window-definitions`:
        await requireMethod(method, "POST", response, async () => {
          await requireService(services.upsertWindowDefinition, response, async (service) => {
            await handleIdempotentPost(
              request,
              response,
              service,
              idempotency,
              "window-definitions",
              maxBodyBytes,
              maxIdempotencyKeyLength,
            );
          });
        });
        return;

      case `${API_PREFIX}/manual-observations`:
        await requireMethod(method, "POST", response, async () => {
          await requireService(services.recordManualObservation, response, async (service) => {
            await handleIdempotentPost(
              request,
              response,
              service,
              idempotency,
              "manual-observations",
              maxBodyBytes,
              maxIdempotencyKeyLength,
            );
          });
        });
        return;

      case `${API_PREFIX}/rebuild-preview`:
        await requireMethod(method, "POST", response, async () => {
          await requireService(services.rebuildPreview, response, async (service) => {
            await handleIdempotentPost(
              request,
              response,
              service,
              idempotency,
              "rebuild-preview",
              maxBodyBytes,
              maxIdempotencyKeyLength,
            );
          });
        });
        return;

      case `${API_PREFIX}/rebuild-apply`:
        await requireMethod(method, "POST", response, async () => {
          await requireService(services.rebuildApply, response, async (service) => {
            await handleIdempotentPost(
              request,
              response,
              service,
              idempotency,
              "rebuild-apply",
              maxBodyBytes,
              maxIdempotencyKeyLength,
            );
          });
        });
        return;

      default:
        sendError(response, HTTP_STATUS.notFound, "not_found", "Not found");
        return;
    }
  }
}

/** Require a specific HTTP method. */
async function requireMethod(
  method: string,
  expected: string,
  response: ServerResponse,
  next: () => Promise<void>,
): Promise<void> {
  if (method !== expected) {
    sendError(
      response,
      HTTP_STATUS.methodNotAllowed,
      "invalid_request",
      `Method ${method} not allowed; use ${expected}`,
    );
    return;
  }
  await next();
}

/** Require an injected service; otherwise return explicit 503. */
async function requireService<T>(
  service: T | undefined,
  response: ServerResponse,
  next: (service: T) => Promise<void>,
): Promise<void> {
  if (service === undefined) {
    sendError(
      response,
      HTTP_STATUS.unavailable,
      "unavailable",
      UNAVAILABLE_REASON,
    );
    return;
  }
  await next(service);
}

/** Handle health endpoints with minimal semantics. */
async function handleHealth(
  response: ServerResponse,
  service: (() => Promise<unknown>) | undefined,
  defaultPayload: { status: string },
): Promise<void> {
  if (service === undefined) {
    sendJson(response, HTTP_STATUS.ok, defaultPayload);
    return;
  }
  sendJson(response, HTTP_STATUS.ok, await service());
}

/** Handle an idempotent POST route. */
async function handleIdempotentPost<T extends {
  account?: string | undefined;
  idempotencyKey?: string | undefined;
}>(
  request: IncomingMessage,
  response: ServerResponse,
  service: (input: T) => Promise<unknown>,
  idempotency: IdempotencyCache,
  route: string,
  maxBodyBytes: number,
  maxIdempotencyKeyLength: number,
): Promise<void> {
  const body = await readJsonBody(request, maxBodyBytes);
  const idempotencyKey = optionalIdempotencyKey(
    typeof body["idempotency_key"] === "string"
      ? body["idempotency_key"]
      : undefined,
    maxIdempotencyKeyLength,
  );

  const account = optionalAccountId(
    typeof body["account"] === "string" ? body["account"] : undefined,
  );

  // Snapshot and fingerprint before invoking a callback. A service may mutate
  // nested input objects, so replay identity must not depend on that mutation.
  const requestSnapshot = immutableSnapshot(body);
  const reservation =
    idempotencyKey === undefined
      ? undefined
      : idempotency.reserve(
          route,
          idempotencyKey,
          fingerprintRequest(route, requestSnapshot),
        );

  if (reservation?.kind === "conflict") {
    sendError(
      response,
      HTTP_STATUS.conflict,
      "conflict",
      "Idempotency-Key replayed with a different request body",
    );
    return;
  }
  if (reservation?.kind === "capacity") {
    sendError(
      response,
      HTTP_STATUS.unavailable,
      "unavailable",
      "Idempotency cache is full",
    );
    return;
  }
  if (reservation?.kind === "replay") {
    sendJson(response, reservation.entry.status, reservation.entry.body);
    return;
  }
  if (reservation?.kind === "pending") {
    const cached = await reservation.promise;
    sendJson(response, cached.status, cached.body);
    return;
  }

  // Give the callback its own mutable copy so nested callback changes cannot
  // alter the request snapshot used by the reservation.
  const payload = {
    ...cloneValue(requestSnapshot),
    account,
    idempotencyKey,
  } as T;

  try {
    const result = await service(payload);
    const status = HTTP_STATUS.accepted;
    if (reservation?.kind === "new") {
      const cached = reservation.complete(status, result);
      sendJson(response, cached.status, cached.body);
      return;
    }
    sendJson(response, status, result);
  } catch (error) {
    if (reservation?.kind === "new") {
      reservation.fail(error);
    }
    throw error;
  }
}

/** Close the server gracefully. */
function closeServer(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close((error) => {
      if (error !== undefined) {
        reject(error);
      } else {
        resolve();
      }
    });
  });
}
