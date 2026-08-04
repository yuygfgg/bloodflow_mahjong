import assert from "node:assert/strict";
import { before, test } from "node:test";
import { isLegalAction, legalActionIdsFromMask, readMaskWords } from "../src/legal.ts";
import { loadWasm } from "./harness.ts";

before(async () => {
  // legal.ts reads ACTION_SPACE_SIZE from the live WASM module.
  await loadWasm();
});

test("legalActionIdsFromMask expands both words", () => {
  const low = 1n << 3n;
  const high = 1n << 0n;
  assert.deepEqual(legalActionIdsFromMask(low, high), [3, 64]);
});

test("isLegalAction checks bounds and bits", () => {
  const low = 1n << 57n;
  const high = 0n;
  assert.equal(isLegalAction(57, low, high), true);
  assert.equal(isLegalAction(0, low, high), false);
  assert.equal(isLegalAction(-1, low, high), false);
  assert.equal(isLegalAction(200, low, high), false);
});

test("readMaskWords accepts array-like mask returns", () => {
  assert.deepEqual(readMaskWords(undefined), [0n, 0n]);
  assert.deepEqual(readMaskWords([3n, 5n]), [3n, 5n]);
  assert.deepEqual(readMaskWords([3, 5]), [3n, 5n]);
});
