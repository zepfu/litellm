/**
 * Authentication, Host/Origin validation, and remote-bind policy for the
 * local API. Token comparison is constant-time; tokens never appear in
 * responses or logs.
 */

import { createHmac, timingSafeEqual } from "node:crypto";
import type { IncomingMessage } from "node:http";

import { LOOPBACK_HOSTS } from "./types.js";

/** Check whether a host (without port) is a loopback address. */
export function isLoopbackHost(host: string): boolean {
  const normalized = host.trim().toLowerCase();
  if (LOOPBACK_HOSTS.has(normalized)) {
    return true;
  }
  // Accept 127.0.0.0/8
  if (/^127(?:\.\d{1,3}){3}$/.test(normalized)) {
    return true;
  }
  // Accept IPv6 loopback with or without brackets
  if (normalized === "::1" || normalized === "[::1]") {
    return true;
  }
  return false;
}

/** Split a Host header value into hostname and port. */
export function parseHostHeader(
  value: string,
): { hostname: string; port: string | null } | null {
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    return null;
  }
  // IPv6 literal in brackets
  const ipv6Match = /^\[(?<host>[^\]]+)\](?::(?<port>\d+))?$/.exec(trimmed);
  if (ipv6Match?.groups) {
    return {
      hostname: ipv6Match.groups.host ?? "",
      port: ipv6Match.groups.port ?? null,
    };
  }
  const lastColon = trimmed.lastIndexOf(":");
  if (lastColon > 0 && /^\d+$/.test(trimmed.slice(lastColon + 1))) {
    return {
      hostname: trimmed.slice(0, lastColon),
      port: trimmed.slice(lastColon + 1),
    };
  }
  return { hostname: trimmed, port: null };
}

/** Constant-time token comparison. */
export function tokenMatches(provided: string, expected: string): boolean {
  if (provided.length === 0 || expected.length === 0) {
    return false;
  }
  // Hash both sides so lengths do not leak through timing.
  const a = createHmac("sha256", "usage-capture-local-api").update(provided).digest();
  const b = createHmac("sha256", "usage-capture-local-api").update(expected).digest();
  return timingSafeEqual(a, b);
}

/** Extract the bearer token from an Authorization header. */
export function extractBearerToken(
  value: string | string[] | undefined,
): string | null {
  const header = Array.isArray(value) ? value[0] : value;
  if (header === undefined) {
    return null;
  }
  const match = /^Bearer\s+(\S+)$/i.exec(header.trim());
  return match ? (match[1] ?? null) : null;
}

/** Validate that a request is authenticated. */
export function isAuthenticated(
  request: IncomingMessage,
  expectedToken: string,
): boolean {
  const token = extractBearerToken(request.headers.authorization);
  if (token === null) {
    return false;
  }
  return tokenMatches(token, expectedToken);
}

/** Validate the Host header against the allowlist. */
export function isHostAllowed(
  hostHeader: string | string[] | undefined,
  boundHost: string,
  allowedHosts: readonly string[] | undefined,
): boolean {
  const header = Array.isArray(hostHeader) ? hostHeader[0] : hostHeader;
  if (header === undefined) {
    return false;
  }
  const parsed = parseHostHeader(header);
  if (parsed === null) {
    return false;
  }
  const hostname = parsed.hostname.toLowerCase();
  if (allowedHosts !== undefined) {
    return allowedHosts.some((allowed) => allowed.toLowerCase() === hostname);
  }
  return isLoopbackHost(hostname) || hostname === boundHost.toLowerCase();
}

/** Validate Origin for browser requests (CSRF protection). */
export function isOriginAllowed(
  originHeader: string | string[] | undefined,
  boundHost: string,
  boundPort: number,
  allowedOrigins: readonly string[] | undefined,
): boolean {
  const header = Array.isArray(originHeader) ? originHeader[0] : originHeader;
  if (header === undefined) {
    // Non-browser clients do not send Origin; they still need the token.
    return true;
  }
  let url: URL;
  try {
    url = new URL(header);
  } catch {
    return false;
  }
  const originHost = url.hostname.toLowerCase();
  if (allowedOrigins !== undefined) {
    return allowedOrigins.some((allowed) => allowed.toLowerCase() === header.toLowerCase());
  }
  // Default: same-origin loopback only.
  return (
    isLoopbackHost(originHost) &&
    (url.port === String(boundPort) || url.port === "")
  );
}

/** Validate that a mutating request carries an explicit CSRF header. */
export function hasCsrfHeader(request: IncomingMessage, token: string): boolean {
  const header = request.headers["x-usage-capture-token"];
  const value = Array.isArray(header) ? header[0] : header;
  if (value === undefined) {
    return false;
  }
  return tokenMatches(value, token);
}
