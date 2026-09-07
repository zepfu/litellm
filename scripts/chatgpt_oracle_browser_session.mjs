#!/usr/bin/env node

import { constants as fsConstants } from "node:fs";
import {
  access,
  chmod,
  mkdtemp,
  readdir,
  readFile,
  rm,
  stat,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const DEFAULT_STARTUP_TIMEOUT_MS = 45_000;
const MAX_STARTUP_TIMEOUT_MS = 120_000;
const MAX_PROTOCOL_LINE_BYTES = 1024;
const SCRATCH_PREFIX = "aawm-oracle-browser-";
const NOOP_LOGGER = () => {};
const SIGNALS = ["SIGINT", "SIGTERM", "SIGQUIT"];

class StartupTimeoutError extends Error {
  constructor() {
    super("startup timeout");
    this.name = "StartupTimeoutError";
  }
}

class StartupAbortedError extends Error {
  constructor() {
    super("startup aborted");
    this.name = "StartupAbortedError";
  }
}

function usage() {
  return [
    "Usage: node scripts/chatgpt_oracle_browser_session.mjs",
    "  --oracle-package-dir <path>",
    "  --chrome-executable <path>",
    "  --base-profile <path>",
    "  [--profile-directory <name-or-direct-child-path>]",
    "  [--headless]",
    "  [--startup-timeout-ms <milliseconds>]",
  ].join("\n");
}

function parseArgs(argv) {
  const values = {
    oraclePackageDir: null,
    chromeExecutable: null,
    baseProfile: null,
    profileDirectory: null,
    headless: false,
    startupTimeoutMs: DEFAULT_STARTUP_TIMEOUT_MS,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--help" || argument === "-h") {
      return { help: true };
    }
    if (argument === "--headless") {
      values.headless = true;
      continue;
    }

    const option = readOption(argv, index);
    if (option === null) {
      throw new Error("unsupported argument");
    }
    index = option.nextIndex;

    if (option.name === "--oracle-package-dir") {
      values.oraclePackageDir = requireOptionValue(option.value);
    } else if (option.name === "--chrome-executable") {
      values.chromeExecutable = requireOptionValue(option.value);
    } else if (option.name === "--base-profile") {
      values.baseProfile = requireOptionValue(option.value);
    } else if (option.name === "--profile-directory") {
      values.profileDirectory = requireOptionValue(option.value);
    } else if (option.name === "--startup-timeout-ms") {
      values.startupTimeoutMs = parseStartupTimeout(option.value);
    } else {
      throw new Error("unsupported argument");
    }
  }

  if (
    !values.oraclePackageDir ||
    !values.chromeExecutable ||
    !values.baseProfile
  ) {
    throw new Error("missing required argument");
  }
  return values;
}

function readOption(argv, index) {
  const argument = argv[index];
  const optionNames = [
    "--oracle-package-dir",
    "--chrome-executable",
    "--base-profile",
    "--profile-directory",
    "--startup-timeout-ms",
  ];
  for (const name of optionNames) {
    if (argument === name) {
      if (index + 1 >= argv.length) {
        throw new Error("missing option value");
      }
      return { name, value: argv[index + 1], nextIndex: index + 1 };
    }
    const prefix = `${name}=`;
    if (argument.startsWith(prefix)) {
      return { name, value: argument.slice(prefix.length), nextIndex: index };
    }
  }
  return null;
}

function requireOptionValue(value) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error("empty option value");
  }
  return value.trim();
}

function parseStartupTimeout(value) {
  const raw = requireOptionValue(value);
  if (!/^\d+$/.test(raw)) {
    throw new Error("invalid startup timeout");
  }
  const parsed = Number(raw);
  if (
    !Number.isSafeInteger(parsed) ||
    parsed < 1_000 ||
    parsed > MAX_STARTUP_TIMEOUT_MS
  ) {
    throw new Error("invalid startup timeout");
  }
  return parsed;
}

async function validateInputs(args) {
  const oraclePackageDir = await resolveDirectory(
    args.oraclePackageDir,
    "oracle package directory",
  );
  const baseProfile = await resolveDirectory(args.baseProfile, "base profile");
  const chromeExecutable = await resolveFile(
    args.chromeExecutable,
    "Chrome executable",
  );
  const profileDirectory = resolveProfileDirectory(
    baseProfile,
    args.profileDirectory,
  );
  return {
    oraclePackageDir,
    chromeExecutable,
    baseProfile,
    profileDirectory,
  };
}

async function resolveDirectory(value, label) {
  const resolved = path.resolve(value);
  try {
    const details = await stat(resolved);
    if (!details.isDirectory()) {
      throw new Error(`${label} is not a directory`);
    }
  } catch {
    throw new Error(`invalid ${label}`);
  }
  return resolved;
}

async function resolveFile(value, label) {
  const resolved = path.resolve(value);
  try {
    const details = await stat(resolved);
    if (!details.isFile()) {
      throw new Error(`${label} is not a file`);
    }
    await access(resolved, fsConstants.X_OK);
  } catch {
    throw new Error(`invalid ${label}`);
  }
  return resolved;
}

function resolveProfileDirectory(baseProfile, requestedProfile) {
  if (requestedProfile === null) {
    return undefined;
  }

  const resolvedBase = path.resolve(baseProfile);
  const candidate = path.resolve(resolvedBase, requestedProfile);
  if (path.dirname(candidate) !== resolvedBase) {
    throw new Error("profile directory must be a direct child of base profile");
  }
  return path.basename(candidate);
}

async function importOracleHelpers(oraclePackageDir) {
  const profileCopyUrl = pathToFileURL(
    path.join(oraclePackageDir, "dist", "src", "browser", "profileCopy.js"),
  ).href;
  const chromeLifecycleUrl = pathToFileURL(
    path.join(
      oraclePackageDir,
      "dist",
      "src",
      "browser",
      "chromeLifecycle.js",
    ),
  ).href;

  const [profileCopy, chromeLifecycle] = await Promise.all([
    import(profileCopyUrl),
    import(chromeLifecycleUrl),
  ]);
  const requiredHelpers = [
    profileCopy.copyChromeProfile,
    chromeLifecycle.launchChrome,
    chromeLifecycle.connectWithNewTab,
    chromeLifecycle.closeTab,
    chromeLifecycle.registerTerminationHooks,
  ];
  if (requiredHelpers.some((helper) => typeof helper !== "function")) {
    throw new Error("Oracle browser helpers are unavailable");
  }
  return {
    copyChromeProfile: profileCopy.copyChromeProfile,
    launchChrome: chromeLifecycle.launchChrome,
    connectWithNewTab: chromeLifecycle.connectWithNewTab,
    closeTab: chromeLifecycle.closeTab,
    registerTerminationHooks: chromeLifecycle.registerTerminationHooks,
  };
}

async function createSession(args, state) {
  const helpers = await importOracleHelpers(args.oraclePackageDir);
  state.helpers = helpers;
  throwIfCleanupRequested(state);

  state.scratchDir = await mkdtemp(path.join(os.tmpdir(), SCRATCH_PREFIX));
  await chmod(state.scratchDir, 0o700);
  throwIfCleanupRequested(state);

  const copiedProfileDirectory = await helpers.copyChromeProfile(
    args.baseProfile,
    state.scratchDir,
    args.profileDirectory,
  );
  throwIfCleanupRequested(state);

  state.chrome = await helpers.launchChrome(
    {
      copyProfileSource: args.baseProfile,
      chromePath: args.chromeExecutable,
      chromeProfile: copiedProfileDirectory,
      debugPort: 0,
      headless: args.headless,
      hideWindow: false,
    },
    state.scratchDir,
    NOOP_LOGGER,
  );
  state.chromeHost = state.chrome.host ?? "127.0.0.1";
  state.chromePort = state.chrome.port;
  if (
    typeof state.chromeHost !== "string" ||
    !state.chromeHost ||
    !Number.isInteger(state.chromePort) ||
    state.chromePort <= 0 ||
    state.chromePort > 65_535
  ) {
    throw new Error("Chrome did not provide a valid CDP endpoint");
  }
  throwIfCleanupRequested(state);

  const connection = await helpers.connectWithNewTab(
    state.chromePort,
    NOOP_LOGGER,
    "about:blank",
    state.chromeHost,
    {
      fallbackToDefault: false,
      retries: 0,
    },
  );
  state.client = connection.client;
  state.pageTargetId = connection.targetId;
  if (typeof state.pageTargetId !== "string" || !state.pageTargetId) {
    throw new Error("Chrome did not provide an owned page target");
  }
  throwIfCleanupRequested(state);

  return {
    cdp_endpoint: formatCdpEndpoint(state.chromeHost, state.chromePort),
    page_target_id: state.pageTargetId,
  };
}

function throwIfCleanupRequested(state) {
  if (state.cleanupRequested) {
    throw new StartupAbortedError();
  }
}

function formatCdpEndpoint(host, port) {
  const endpointHost =
    host.includes(":") && !host.startsWith("[") ? `[${host}]` : host;
  return `http://${endpointHost}:${port}`;
}

function boundedProtocolLine(session) {
  const line = JSON.stringify({
    cdp_endpoint: session.cdp_endpoint,
    page_target_id: session.page_target_id,
  });
  if (Buffer.byteLength(line, "utf8") > MAX_PROTOCOL_LINE_BYTES) {
    throw new Error("CDP session protocol line is too large");
  }
  return `${line}\n`;
}

async function listOwnedChromePids(scratchDir) {
  if (process.platform !== "linux") {
    return [];
  }
  let entries;
  try {
    entries = await readdir("/proc");
  } catch {
    return [];
  }
  const marker = `--user-data-dir=${scratchDir}`;
  const pids = [];
  for (const entry of entries) {
    if (!/^\d+$/.test(entry)) {
      continue;
    }
    let commandLine;
    try {
      commandLine = await readFile(path.join("/proc", entry, "cmdline"));
    } catch {
      continue;
    }
    if (commandLine.toString("utf8").split("\0").includes(marker)) {
      pids.push(Number(entry));
    }
  }
  return pids;
}

async function signalOwnedChromePid(pid, scratchDir, signal) {
  if (process.platform !== "linux" || !Number.isInteger(pid)) {
    return;
  }
  const ownedPids = await listOwnedChromePids(scratchDir);
  if (!ownedPids.includes(pid)) {
    return;
  }
  try {
    process.kill(-pid, signal);
  } catch {
    try {
      process.kill(pid, signal);
    } catch {
      // The owned process may have exited between the evidence check and signal.
    }
  }
}

async function terminateOwnedStartupChrome(scratchDir) {
  if (!scratchDir) {
    return;
  }
  const ownedPids = await listOwnedChromePids(scratchDir);
  for (const pid of ownedPids) {
    await signalOwnedChromePid(pid, scratchDir, "SIGTERM");
  }
  if (ownedPids.length === 0) {
    return;
  }
  await new Promise((resolve) => setTimeout(resolve, 100));
  const remainingPids = await listOwnedChromePids(scratchDir);
  for (const pid of remainingPids) {
    await signalOwnedChromePid(pid, scratchDir, "SIGKILL");
  }
}

async function cleanupOwned(state, helpers) {
  if (state.cleanupPromise) {
    return state.cleanupPromise;
  }

  state.cleanupPromise = (async () => {
    const targetId = state.pageTargetId;
    const client = state.client;
    const chrome = state.chrome;
    const scratchDir = state.scratchDir;

    if (targetId && chrome && Number.isInteger(chrome.port)) {
      await helpers
        .closeTab(
          chrome.port,
          targetId,
          NOOP_LOGGER,
          chrome.host ?? "127.0.0.1",
        )
        .catch(() => undefined);
    }
    if (client && typeof client.close === "function") {
      await client.close().catch(() => undefined);
    }
    if (chrome && typeof chrome.kill === "function") {
      try {
        await Promise.resolve(chrome.kill());
      } catch {
        // Cleanup continues with owned PID evidence below.
      }
    }
    await terminateOwnedStartupChrome(scratchDir);
    if (scratchDir) {
      await rm(scratchDir, { recursive: true, force: true }).catch(
        () => undefined,
      );
    }
    state.pageTargetId = null;
    state.client = null;
    state.chrome = null;
    state.scratchDir = null;
  })();

  try {
    return await state.cleanupPromise;
  } finally {
    state.cleanupPromise = null;
  }
}

function waitForSignal() {
  let settled = false;
  let resolveSignal;
  const promise = new Promise((resolve) => {
    resolveSignal = resolve;
  });
  const handlers = new Map();
  for (const signal of SIGNALS) {
    const handler = () => {
      if (settled) {
        return;
      }
      settled = true;
      resolveSignal(signal);
    };
    handlers.set(signal, handler);
    process.once(signal, handler);
  }
  return {
    promise,
    remove() {
      for (const [signal, handler] of handlers) {
        process.removeListener(signal, handler);
      }
    },
  };
}

function watchStdinEof() {
  let settled = false;
  let closed = false;
  let resolveEof;
  const promise = new Promise((resolve) => {
    resolveEof = resolve;
  });
  const finish = () => {
    if (settled) {
      return;
    }
    settled = true;
    closed = true;
    process.stdin.removeListener("end", finish);
    process.stdin.removeListener("close", finish);
    process.stdin.removeListener("error", finish);
    resolveEof();
  };
  process.stdin.once("end", finish);
  process.stdin.once("close", finish);
  process.stdin.once("error", finish);
  process.stdin.resume();
  if (process.stdin.readableEnded) {
    finish();
  }
  return {
    promise,
    isClosed() {
      return closed;
    },
    remove() {
      process.stdin.removeListener("end", finish);
      process.stdin.removeListener("close", finish);
      process.stdin.removeListener("error", finish);
    },
  };
}

function watchStdoutErrors() {
  let errorSeen = false;
  let resolveError;
  const promise = new Promise((resolve) => {
    resolveError = resolve;
  });
  const handleError = () => {
    if (errorSeen) {
      return;
    }
    errorSeen = true;
    resolveError();
  };
  process.stdout.on("error", handleError);
  return {
    promise,
    hasError() {
      return errorSeen;
    },
    remove() {
      process.stdout.removeListener("error", handleError);
    },
  };
}

function writeProtocolLine(line) {
  return new Promise((resolve, reject) => {
    process.stdout.write(line, "utf8", (error) => {
      if (error) {
        reject(error);
        return;
      }
      resolve();
    });
  });
}

function sanitizedErrorName(error) {
  const name = error instanceof Error ? error.name : "Error";
  return /^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(name) ? name : "Error";
}

function reportFailure(stage, error) {
  process.stderr.write(
    `oracle browser session ${stage} failed (${sanitizedErrorName(error)})\n`,
  );
}

async function settleStartup(state) {
  if (!state.startupPromise) {
    return;
  }
  try {
    await state.startupPromise;
  } catch {
    // The caller receives only the bounded, sanitized failure below.
  }
}

async function run() {
  let args;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (error) {
    reportFailure("argument validation", error);
    process.stderr.write(`${usage()}\n`);
    return 2;
  }
  if (args.help) {
    process.stdout.write(`${usage()}\n`);
    return 0;
  }

  const state = {
    cleanupRequested: false,
    cleanupPromise: null,
    startupPromise: null,
    helpers: null,
    scratchDir: null,
    chrome: null,
    chromeHost: null,
    chromePort: null,
    client: null,
    pageTargetId: null,
  };
  const signalWait = waitForSignal();
  const stdinWait = watchStdinEof();
  const stdoutErrors = watchStdoutErrors();
  let removeTerminationHooks = null;

  try {
    const validated = await validateInputs(args);
    args = { ...args, ...validated };

    state.startupPromise = createSession(args, state);
    let startupTimer;
    const startupResult = await Promise.race([
      state.startupPromise.then(
        (session) => ({ kind: "ready", session }),
        (error) => ({ kind: "error", error }),
      ),
      signalWait.promise.then((signal) => ({ kind: "signal", signal })),
      stdinWait.promise.then(() => ({ kind: "stdin-eof" })),
      stdoutErrors.promise.then(() => ({ kind: "stdout-error" })),
      new Promise((resolve) => {
        startupTimer = setTimeout(
          () => resolve({ kind: "timeout" }),
          args.startupTimeoutMs,
        );
      }),
    ]);
    clearTimeout(startupTimer);

    if (startupResult.kind === "signal") {
      signalWait.remove();
      state.cleanupRequested = true;
      await cleanupWithAvailableHelpers(state);
      process.exitCode = startupResult.signal === "SIGINT" ? 130 : 143;
      return process.exitCode;
    }
    if (startupResult.kind === "timeout") {
      signalWait.remove();
      state.cleanupRequested = true;
      await cleanupWithAvailableHelpers(state);
      reportFailure("startup", new StartupTimeoutError());
      return 1;
    }
    if (
      startupResult.kind === "stdin-eof" ||
      startupResult.kind === "stdout-error"
    ) {
      signalWait.remove();
      state.cleanupRequested = true;
      await cleanupWithAvailableHelpers(state);
      reportFailure("startup", new StartupAbortedError());
      return 1;
    }
    if (startupResult.kind === "error") {
      signalWait.remove();
      state.cleanupRequested = true;
      await cleanupWithAvailableHelpers(state);
      reportFailure("startup", startupResult.error);
      return 1;
    }

    const helpers = state.helpers;
    if (!helpers) {
      throw new Error("Oracle browser helpers are unavailable");
    }
    removeTerminationHooks = helpers.registerTerminationHooks(
      state.chrome,
      state.scratchDir,
      false,
      NOOP_LOGGER,
      {
        forceProfileCleanup: true,
        preserveUserDataDir: false,
      },
    );
    signalWait.remove();

    if (stdinWait.isClosed() || stdoutErrors.hasError()) {
      removeTerminationHooks();
      removeTerminationHooks = null;
      state.cleanupRequested = true;
      await cleanupWithAvailableHelpers(state);
      reportFailure("startup", new StartupAbortedError());
      return 1;
    }
    try {
      await writeProtocolLine(boundedProtocolLine(startupResult.session));
    } catch {
      removeTerminationHooks();
      removeTerminationHooks = null;
      state.cleanupRequested = true;
      await cleanupWithAvailableHelpers(state);
      reportFailure("protocol", new StartupAbortedError());
      return 1;
    }

    const captureResult = await Promise.race([
      stdinWait.promise.then(() => "stdin-eof"),
      stdoutErrors.promise.then(() => "stdout-error"),
    ]);
    removeTerminationHooks();
    removeTerminationHooks = null;
    state.cleanupRequested = true;
    await cleanupWithAvailableHelpers(state);
    if (captureResult === "stdout-error") {
      reportFailure("protocol", new StartupAbortedError());
      return 1;
    }
    return 0;
  } catch (error) {
    signalWait.remove();
    if (removeTerminationHooks) {
      removeTerminationHooks();
      removeTerminationHooks = null;
    }
    state.cleanupRequested = true;
    await cleanupWithAvailableHelpers(state);
    reportFailure("startup", error);
    return 1;
  } finally {
    stdinWait.remove();
    stdoutErrors.remove();
  }
}

async function cleanupWithAvailableHelpers(state) {
  await settleStartup(state);
  if (state.helpers) {
    await cleanupOwned(state, state.helpers);
    return;
  }
  if (state.scratchDir) {
    const scratchDir = state.scratchDir;
    state.scratchDir = null;
    await rm(scratchDir, { recursive: true, force: true }).catch(
      () => undefined,
    );
  }
}

const exitCode = await run();
if (exitCode !== 0) {
  process.exitCode = exitCode;
}
