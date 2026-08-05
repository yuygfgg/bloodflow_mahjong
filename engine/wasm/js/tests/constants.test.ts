import assert from "node:assert/strict";
import { before, test } from "node:test";
import { loadWasm, wasmConst, type WasmModule } from "./harness.ts";

let wasm: WasmModule;

before(async () => {
  wasm = await loadWasm();
});

/** Collect zero-arg uppercase numeric getters from the live WASM module. */
function exportedConstants(mod: WasmModule): Record<string, number> {
  const out: Record<string, number> = {};
  for (const name of Object.keys(mod)) {
    if (name !== name.toUpperCase()) {
      continue;
    }
    const value = mod[name];
    if (typeof value !== "function") {
      continue;
    }
    // Skip class constructors and multi-arg helpers.
    if ((value as Function).length !== 0) {
      continue;
    }
    try {
      const result = (value as () => unknown)();
      if (typeof result === "number") {
        out[name] = result;
      }
    } catch {
      // Not a constant getter.
    }
  }
  return out;
}

test("WASM module exports a full engine constant table", () => {
  const constants = exportedConstants(wasm);
  const names = Object.keys(constants);
  assert.ok(names.length > 40, `expected a full constant table, got ${names.length}`);
  assert.equal(constants.ENGINE_RULES_VERSION, 7);
  for (const name of names) {
    assert.equal(wasmConst(wasm, name), constants[name], name);
  }
});

test("action layout matches the fixed policy contract", () => {
  assert.deepEqual(
    [
      wasmConst(wasm, "ACTION_EXCHANGE_TILE_OFFSET"),
      wasmConst(wasm, "ACTION_CHOOSE_MISSING_OFFSET"),
      wasmConst(wasm, "ACTION_DISCARD_OFFSET"),
      wasmConst(wasm, "ACTION_HU"),
      wasmConst(wasm, "ACTION_PONG"),
      wasmConst(wasm, "ACTION_EXPOSED_KONG"),
      wasmConst(wasm, "ACTION_CONCEALED_KONG_OFFSET"),
      wasmConst(wasm, "ACTION_ADDED_KONG_OFFSET"),
      wasmConst(wasm, "ACTION_PASS"),
    ],
    [0, 27, 30, 57, 58, 59, 60, 87, 114],
  );
});

test("UI summary buffers do not change the fixed event record", () => {
  assert.equal(wasmConst(wasm, "EVENT_RECORD_WIDTH"), 8);
  assert.equal(wasmConst(wasm, "PLAYER_UI_STATS_FIELDS"), 5);
  assert.equal(wasmConst(wasm, "PLAYER_UI_STATS_WIDTH"), 20);
  assert.equal(wasmConst(wasm, "WALL_SETTLEMENT_FIELDS"), 3);
  assert.equal(wasmConst(wasm, "WALL_SETTLEMENT_META_WIDTH"), 12);
  assert.equal(wasmConst(wasm, "WALL_SETTLEMENT_HANDS_WIDTH"), 108);
  assert.equal(wasmConst(wasm, "EVENT_KIND_SETTLEMENT_STAGE"), 11);
  assert.notEqual(
    wasmConst(wasm, "SETTLEMENT_STAGE_FLOWER_PIG"),
    wasmConst(wasm, "SETTLEMENT_STAGE_DAJIAO"),
  );
});
