import { DEFAULT_REFRESH_INTERVAL, SchedulerTransitionError } from "./types.js";

export interface ParsedRefreshInterval {
  readonly spec: string;
  readonly milliseconds: number;
}

const INTERVAL_PATTERN = /^PT(?:(\d+)H)?(?:(\d+)M)?$/;
const MINIMUM_INTERVAL_MS = 60_000;

export function parseRefreshInterval(
  value: string = DEFAULT_REFRESH_INTERVAL,
): ParsedRefreshInterval {
  const match = INTERVAL_PATTERN.exec(value);
  const hours = match?.[1] === undefined ? 0 : Number(match[1]);
  const minutes = match?.[2] === undefined ? 0 : Number(match[2]);
  if (
    !match ||
    (match[1] === undefined && match[2] === undefined) ||
    !Number.isSafeInteger(hours) ||
    !Number.isSafeInteger(minutes)
  ) {
    throw new SchedulerTransitionError(
      "invalid_request",
      "interval must use PT hours/minutes syntax, for example PT1H",
    );
  }

  const milliseconds = hours * 3_600_000 + minutes * 60_000;
  if (milliseconds < MINIMUM_INTERVAL_MS) {
    throw new SchedulerTransitionError(
      "invalid_request",
      "interval must be at least one minute",
    );
  }
  return { spec: formatInterval(milliseconds), milliseconds };
}

export function formatInterval(milliseconds: number): string {
  if (!Number.isSafeInteger(milliseconds) || milliseconds < MINIMUM_INTERVAL_MS) {
    throw new SchedulerTransitionError(
      "invalid_request",
      "interval must be a safe integer of at least one minute",
    );
  }
  const hours = Math.floor(milliseconds / 3_600_000);
  const minutes = Math.floor((milliseconds % 3_600_000) / 60_000);
  return `PT${hours > 0 ? `${hours}H` : ""}${minutes > 0 ? `${minutes}M` : ""}`;
}
