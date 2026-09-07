// Flat ESLint config for Stage 1. Core rules only; the TypeScript parser keeps
// lint syntax-aware while TypeScript checks remain delegated to typecheck.
import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default [
  {
    ignores: ["dist/**", "node_modules/**", "coverage/**", "state/**"],
  },
  js.configs.recommended,
  {
    files: ["bin/**/*.mjs", "eslint.config.js"],
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: "module",
      globals: {
        console: "readonly",
        process: "readonly",
      },
    },
  },
  {
    files: ["src/**/*.ts", "tests/**/*.ts", "vitest.config.ts"],
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: "module",
      parser: tseslint.parser,
    },
    rules: {
      // TypeScript performs definite-assignment and undefined checks; the core
      // no-undef rule is not type-aware and produces false positives on TS.
      "no-undef": "off",
      "no-unused-vars": "off",
    },
  },
];
