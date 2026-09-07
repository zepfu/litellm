import {
  DEFAULT_REFRESH_INTERVAL,
} from "./types.js";

export interface ParsedRefreshInterval {
  spec: string;
  minutes: number;
  milliseconds: number;
}

const INTERVAL_PATTERN = /^PT(?:(\d+)H)?(?:(\d+)M)?$/;
const MINIMUM_INTERVAL_MINUTES = 5;

export function parseRefreshInterval(
  value: string = DEFAULT_REFRESH_INTERVAL,
): ParsedRefreshInterval {
  const match = INTERVAL_PATTERN.exec(value);
  if (!match || (match[1] === undefined && match[2] === undefined)) {
    throw new Error(
      "refresh interval must use PT hours/minutes syntax, for example PT5M, PT1H, or PT1H30M",
    );
  }

  const hours = match[1] === undefined ? 0 : Number(match[1]);
  const minutes = match[2] === undefined ? 0 : Number(match[2]);
  const totalMinutes = hours * 60 + minutes;
  if (
    !Number.isSafeInteger(hours) ||
    !Number.isSafeInteger(minutes) ||
    !Number.isSafeInteger(totalMinutes) ||
    totalMinutes < MINIMUM_INTERVAL_MINUTES
  ) {
    throw new Error(
      `refresh interval must be at least ${MINIMUM_INTERVAL_MINUTES} minutes`,
    );
  }

  return {
    spec: formatRefreshInterval(totalMinutes),
    minutes: totalMinutes,
    milliseconds: totalMinutes * 60_000,
  };
}

export function formatRefreshInterval(totalMinutes: number): string {
  if (!Number.isSafeInteger(totalMinutes) || totalMinutes < MINIMUM_INTERVAL_MINUTES) {
    throw new Error(
      `refresh interval must be at least ${MINIMUM_INTERVAL_MINUTES} minutes`,
    );
  }
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return `PT${hours > 0 ? `${hours}H` : ""}${minutes > 0 ? `${minutes}M` : ""}`;
}
