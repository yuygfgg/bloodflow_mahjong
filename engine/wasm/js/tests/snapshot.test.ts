import assert from "node:assert/strict";
import { before, test } from "node:test";
import {
  PHASE_EXCHANGE,
  PHASE_FINISHED,
  PLAYER_COUNT,
  TILE_KIND_COUNT,
} from "../../pkg/bloodflow_mahjong_wasm.js";
import {
  UI_SNAPSHOT_SCHEMA_VERSION,
  buildUiSnapshot,
  createObservationBuffers,
} from "../src/snapshot.ts";
import {
  loadWasm,
  playUntil,
  simpleRulePolicy,
  type WasmModule,
} from "./harness.ts";

let wasm: WasmModule;

before(async () => {
  wasm = await loadWasm();
});

test("buildUiSnapshot is viewer-relative and schema-complete", () => {
  const game = new wasm.Game(42n);
  playUntil(game, simpleRulePolicy, {
    maxSteps: 32,
    stopWhen: (g) => g.phase !== PHASE_EXCHANGE(),
  });

  const buffers = createObservationBuffers();
  for (const viewer of [0, 1, 2, 3]) {
    const snapshot = buildUiSnapshot(game, {
      viewer,
      engineRulesVersion: 6,
      buffers,
    });

    assert.equal(snapshot.schemaVersion, UI_SNAPSHOT_SCHEMA_VERSION);
    assert.equal(snapshot.engineRulesVersion, 6);
    assert.equal(snapshot.ownHand.length, TILE_KIND_COUNT());
    assert.equal(snapshot.unlockedHand.length, TILE_KIND_COUNT());
    assert.equal(snapshot.lockedTiles.length, PLAYER_COUNT());
    assert.equal(snapshot.scores.length, PLAYER_COUNT());
    assert.equal(snapshot.legalActionMask.length, 2);
    assert.ok(Array.isArray(snapshot.legalActionIds));
    assert.equal(snapshot.dealer, (game.dealer - viewer) & 3);

    if (snapshot.decisionActor != null && game.decision != null) {
      assert.equal(snapshot.decisionActor, (game.decision[0]! - viewer) & 3);
    }

    // Opponent concealed planes are not exposed as full hands.
    assert.equal(snapshot.handCounts.length, 4);
    for (let relative = 1; relative < 4; relative += 1) {
      assert.ok(snapshot.handCounts[relative]! >= 0);
    }
  }
  game.free();
});

test("terminal snapshot includes rankings and termination reason", () => {
  const game = new wasm.Game(42n);
  playUntil(game, simpleRulePolicy);
  const snapshot = buildUiSnapshot(game, {
    viewer: 2,
    engineRulesVersion: 6,
  });
  assert.equal(snapshot.phase, PHASE_FINISHED());
  assert.ok(snapshot.rankings != null);
  assert.equal(snapshot.rankings!.length, 4);
  assert.deepEqual([...snapshot.rankings!].sort(), [0, 1, 2, 3]);
  assert.equal(snapshot.terminationReason, game.terminationReason ?? null);
  assert.equal(snapshot.decisionActor, null);
  game.free();
});

test("snapshot draw tile stays hidden for non-drawer viewers", () => {
  const game = new wasm.Game(127n);
  playUntil(game, simpleRulePolicy, {
    maxSteps: 256,
    stopWhen: (g) => g.currentDraw != null,
  });
  assert.ok(game.currentDraw != null);
  const drawer = game.currentDraw![0]!;
  const tile = game.currentDraw![1]!;
  const other = (drawer + 1) % 4;

  const drawerSnap = buildUiSnapshot(game, {
    viewer: drawer,
    engineRulesVersion: 6,
  });
  const otherSnap = buildUiSnapshot(game, {
    viewer: other,
    engineRulesVersion: 6,
  });
  assert.equal(drawerSnap.drawTile, tile);
  assert.equal(otherSnap.drawTile, -1);
  game.free();
});
