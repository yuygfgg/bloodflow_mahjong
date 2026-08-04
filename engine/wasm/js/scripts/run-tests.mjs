#!/usr/bin/env node
/**
 * Full WASM + TypeScript engine test harness entrypoint.
 *
 * Requires a built package under engine/wasm/pkg (run `../../build.sh`).
 */
import { spawn } from "node:child_process";
import { access, readdir } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const jsRoot = resolve(here, "..");
const pkgDir = resolve(jsRoot, "../pkg");
const testsDir = resolve(jsRoot, "tests");
const pkgWasm = resolve(pkgDir, "bloodflow_mahjong_wasm_bg.wasm");
const pkgJs = resolve(pkgDir, "bloodflow_mahjong_wasm.js");

async function requireArtifacts() {
  for (const path of [pkgWasm, pkgJs]) {
    try {
      await access(path);
    } catch {
      console.error(`Missing artifact: ${path}`);
      console.error("Build first with: ./engine/wasm/build.sh");
      process.exit(1);
    }
  }
}

async function listTestFiles() {
  const names = await readdir(testsDir);
  return names
    .filter((name) => name.endsWith(".test.ts"))
    .sort()
    .map((name) => join("tests", name));
}

function run(command, args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      stdio: "inherit",
      cwd: jsRoot,
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) {
        resolvePromise(undefined);
      } else {
        reject(new Error(`${command} ${args.join(" ")} exited with ${code ?? "null"}`));
      }
    });
  });
}

await requireArtifacts();
const testFiles = await listTestFiles();
if (testFiles.length === 0) {
  console.error(`No test files found under ${testsDir}`);
  process.exit(1);
}

await run("npx", ["tsc", "-p", "tsconfig.json"]);
// Node 24 process isolation collapses each file into one opaque test entry.
// Disable isolation so individual `test()` cases report and share one WASM load.
await run(process.execPath, [
  "--experimental-strip-types",
  "--test",
  "--test-isolation=none",
  "--test-reporter",
  "spec",
  ...testFiles,
]);
console.log(`All engine WASM harness tests passed (${testFiles.length} files).`);
