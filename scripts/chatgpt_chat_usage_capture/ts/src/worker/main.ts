import { createInterface } from "node:readline";
import {
  DEFAULT_WORKER_BOUNDS,
  MAX_REQUEST_ID_LENGTH,
  WORKER_PROTOCOL_VERSION,
  type CollectorRunEnvelope,
  type CommitPagePayload,
  type OpaquePageRequest,
  type WorkerBridge,
  type WorkerOperation,
  type WorkerRequest,
  type WorkerResponse,
} from "./contracts.js";
import { BoundedWorkerError } from "./collect.js";

export interface WorkerDependencies {
  bridge: WorkerBridge;
  envelope: CollectorRunEnvelope;
  maxFrameBytes?: number;
  maxTotalBytes?: number;
  maxRequests?: number;
  deadlineAt?: number;
}

export function createWorkerApplication(deps: WorkerDependencies) {
  let active = true;
  let lastRequest: WorkerRequest | null = null;
  let lastResponse: string | null = null;
  let lastCanonicalPayload: string | null = null;
  let cancelled = false;
  let totalBytes = 0;
  let requestCount = 0;
  const appPublic = {
    cancel: () => {
      cancelled = true;
      active = false;
    },
  };

  const validateLine = (line: string): boolean => {
    const bytes = Buffer.byteLength(line, "utf8") + 1;
    if (bytes > (deps.maxFrameBytes ?? DEFAULT_WORKER_BOUNDS.maxFrameBytes)) {
      return false;
    }
    totalBytes += bytes;
    return totalBytes <= (deps.maxTotalBytes ?? DEFAULT_WORKER_BOUNDS.maxTotalBytes);
  };

  async function handle(request: WorkerRequest): Promise<WorkerResponse> {
    requestCount += 1;
    if (requestCount > (deps.maxRequests ?? DEFAULT_WORKER_BOUNDS.maxRequests)) {
      return fail(request.requestId, "bounds_exceeded", true, true);
    }
    if (deps.deadlineAt !== undefined && Date.now() >= deps.deadlineAt) {
      return fail(request.requestId, "bounds_exceeded", true, true);
    }
    const validation = validateRequest(request, deps.envelope);
    if (validation) {
      return validation;
    }
    if (lastRequest?.requestId === request.requestId) {
      const canonicalPayload = JSON.stringify(request.payload ?? null);
      if (lastCanonicalPayload !== canonicalPayload) {
        return fail(request.requestId, "protocol_invalid", false, true);
      }
      if (lastResponse !== null) {
        return JSON.parse(lastResponse) as WorkerResponse;
      }
      return fail(request.requestId, "protocol_invalid", false, true);
    }

    lastRequest = request;
    lastCanonicalPayload = JSON.stringify(request.payload ?? null);
    lastResponse = null;
    try {
      const result = await dispatch(request.operation, request.payload, deps.bridge, appPublic);
      const response: WorkerResponse = {
        protocolVersion: WORKER_PROTOCOL_VERSION,
        requestId: request.requestId,
        ok: true,
        result,
      };
      lastResponse = JSON.stringify(response);
      return response;
    } catch (error) {
      const bounded = error instanceof BoundedWorkerError;
      const response = fail(
        request.requestId,
        bounded ? error.code : "history_reader_failed",
        bounded ? error.retryable : true,
        bounded ? error.coverageIncomplete : true,
      );
      lastResponse = JSON.stringify(response);
      return response;
    }
  }

  return {
    handle,
    cancel: appPublic.cancel,
    isActive: () => active,
  };
}

export async function serveWorkerLines(
  deps: WorkerDependencies,
  input: NodeJS.ReadableStream,
  output: NodeJS.WritableStream,
): Promise<void> {
  const app = createWorkerApplication(deps);
  const validateLine = (line: string): boolean => {
    const bytes = Buffer.byteLength(line, "utf8") + 1;
    if (bytes > (deps.maxFrameBytes ?? DEFAULT_WORKER_BOUNDS.maxFrameBytes)) {
      return false;
    }
    return true;
  };
  const lines = createInterface({ input, crlfDelay: Infinity });
  for await (const line of lines) {
    if (!app.isActive()) {
      break;
    }
    if (!validateLine(line)) {
      break;
    }
    const request = parseLine(line);
    if (!request) {
      continue;
    }
    const response = await app.handle(request);
    output.write(`${JSON.stringify(response)}\n`);
    if (response.ok && request.operation === "finishRun") {
      break;
    }
  }
}

async function dispatch(
  operation: WorkerOperation,
  payload: unknown,
  bridge: WorkerBridge,
  app?: { cancel(): void },
): Promise<unknown> {
  switch (operation) {
    case "loadState":
    case "loadReportSnapshot":
      requireNoPayload(payload);
      return operation === "loadState" ? bridge.loadState() : bridge.loadReportSnapshot();
    case "compareAndSetState":
      return bridge.compareAndSetState(typed(payload, "state"));
    case "readHistory":
      return bridge.readHistory(typed(payload, "read"));
    case "loadConversationMetadata":
      return bridge.loadConversationMetadata(typed(payload, "metadata").conversationId);
    case "commitPage":
      return bridge.commitPage(typed(payload, "commit"));
    case "finishRun":
      requireNoPayload(payload);
      return bridge.finishRun();
    case "cancel":
      requireNoPayload(payload);
      app?.cancel();
      return { cancelled: true };
    default:
      throw new BoundedWorkerError("operation_unsupported", false, true);
  }
}

function parseLine(line: string): WorkerRequest | null {
  try {
    return JSON.parse(line) as WorkerRequest;
  } catch {
    return null;
  }
}

function validateRequest(
  request: WorkerRequest,
  envelope: CollectorRunEnvelope,
): WorkerResponse | null {
  if (
    request.protocolVersion !== WORKER_PROTOCOL_VERSION ||
    typeof request.requestId !== "string" ||
    request.requestId.length === 0 ||
    request.requestId.length > MAX_REQUEST_ID_LENGTH ||
    request.runId !== envelope.runId ||
    request.collectorAccountId !== envelope.collectorAccountId ||
    request.profileId !== envelope.profileId ||
    request.bindingGeneration !== envelope.bindingGeneration ||
    request.leaseFencingToken !== envelope.leaseFencingToken
  ) {
    const requestId = typeof request.requestId === "string" ? request.requestId : "";
    return fail(
      requestId,
      request.bindingGeneration !== envelope.bindingGeneration ||
        request.leaseFencingToken !== envelope.leaseFencingToken
        ? "fence_invalid"
        : "protocol_invalid",
      false,
      true,
    );
  }
  return null;
}

function requireNoPayload(payload: unknown): void {
  if (payload !== undefined) {
    throw new BoundedWorkerError("protocol_invalid", false, true);
  }
}

function typed<T extends "state" | "read" | "metadata" | "commit">(
  payload: unknown,
  kind: T,
): T extends "state"
  ? Parameters<WorkerBridge["compareAndSetState"]>[0]
  : T extends "read"
    ? OpaquePageRequest
    : T extends "metadata"
      ? { conversationId: string }
      : CommitPagePayload {
  return payload as never;
}

type WorkerErrorCode =
  | "protocol_invalid"
  | "fence_invalid"
  | "state_conflict"
  | "history_contract_unavailable"
  | "history_reader_failed"
  | "bounds_exceeded"
  | "conversation_not_found"
  | "operation_unsupported"
  | "cancelled";

function fail(
  requestId: string,
  code: WorkerErrorCode,
  retryable: boolean,
  coverageIncomplete: boolean,
): WorkerResponse {
  return {
    protocolVersion: WORKER_PROTOCOL_VERSION,
    requestId,
    ok: false,
    error: { code, retryable, coverageIncomplete },
  };
}
