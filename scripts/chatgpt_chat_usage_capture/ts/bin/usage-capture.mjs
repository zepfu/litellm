#!/usr/bin/env node
// Thin launcher for the TypeScript Stage-1 CLI. Runs the compiled output; falls
// back to a helpful error when the project has not been built yet.
import { pathToFileURL, fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const entry = resolve(here, "..", "dist", "src", "cli", "main.js");

if (!existsSync(entry)) {
  process.stderr.write(
    "usage-capture: compiled output not found. Run `npm run build` in scripts/chatgpt_chat_usage_capture/ts first.\n",
  );
  process.exit(2);
}

const mod = await import(pathToFileURL(entry).href);
const code = await mod.run(process.argv.slice(2));
process.exit(code);
