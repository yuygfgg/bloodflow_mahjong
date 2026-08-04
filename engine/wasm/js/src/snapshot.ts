import {
  EVENT_HISTORY_CAPACITY,
  EVENT_RECORD_WIDTH,
  META_OBSERVATION_WIDTH,
  MELD_FIELDS,
  MELD_OBSERVATION_WIDTH,
  MELD_SLOTS,
  PHASE_FINISHED,
  PLAYER_COUNT,
  RIVER_FIELDS,
  RIVER_OBSERVATION_WIDTH,
  RIVER_TILE_CAPACITY,
  TILE_KIND_COUNT,
  TILE_OBSERVATION_WIDTH,
} from "../../pkg/bloodflow_mahjong_wasm.js";
import { legalActionIdsFromMask, readMaskWords } from "./legal.ts";
import type {
  EventRecord,
  GameLike,
  MeldView,
  ObservationBuffers,
  RiverEntry,
  UiSnapshot,
} from "./types.ts";

/** Client snapshot schema version. Engine rules version is separate. */
export const UI_SNAPSHOT_SCHEMA_VERSION = 1;

/** Allocate reusable observation and event buffers for one viewer. */
export function createObservationBuffers(): ObservationBuffers {
  return {
    tileObs: new Uint8Array(TILE_OBSERVATION_WIDTH()),
    melds: new Uint8Array(MELD_OBSERVATION_WIDTH()),
    river: new Uint8Array(RIVER_OBSERVATION_WIDTH()),
    meta: new Int32Array(META_OBSERVATION_WIDTH()),
    events: new Int32Array(EVENT_HISTORY_CAPACITY() * EVENT_RECORD_WIDTH()),
    stepEvents: new Int32Array(EVENT_HISTORY_CAPACITY() * EVENT_RECORD_WIDTH()),
  };
}

function relativeSeat(absolute: number, viewer: number): number {
  return (absolute - viewer) & 3;
}

function relativeQuad<T>(values: ArrayLike<T>, viewer: number): [T, T, T, T] {
  return [
    values[(viewer + 0) & 3] as T,
    values[(viewer + 1) & 3] as T,
    values[(viewer + 2) & 3] as T,
    values[(viewer + 3) & 3] as T,
  ];
}

function copyPlane(tileObs: Uint8Array, plane: number): Uint8Array {
  const start = plane * TILE_KIND_COUNT();
  return tileObs.slice(start, start + TILE_KIND_COUNT());
}

function parseMelds(melds: Uint8Array): [
  MeldView[],
  MeldView[],
  MeldView[],
  MeldView[],
] {
  const bySeat: [MeldView[], MeldView[], MeldView[], MeldView[]] = [[], [], [], []];
  for (let seat = 0; seat < PLAYER_COUNT(); seat += 1) {
    for (let slot = 0; slot < MELD_SLOTS(); slot += 1) {
      const base = (seat * MELD_SLOTS() + slot) * MELD_FIELDS();
      const tile = melds[base] ?? 255;
      if (tile === 255) {
        continue;
      }
      bySeat[seat].push({
        tile,
        kind: melds[base + 1] ?? 0,
        sourceRelative: melds[base + 2] ?? 0,
      });
    }
  }
  return bySeat;
}

function parseRiver(river: Uint8Array, length: number): RiverEntry[] {
  const out: RiverEntry[] = [];
  const n = Math.min(length, RIVER_TILE_CAPACITY());
  for (let i = 0; i < n; i += 1) {
    const base = i * RIVER_FIELDS();
    const tile = river[base] ?? 255;
    if (tile === 255) {
      break;
    }
    out.push({
      tile,
      ownerRelative: river[base + 1] ?? 0,
    });
  }
  return out;
}

function parseEvents(buffer: Int32Array, count: number): EventRecord[] {
  const out: EventRecord[] = [];
  for (let i = 0; i < count; i += 1) {
    const base = i * EVENT_RECORD_WIDTH();
    out.push([
      buffer[base] ?? -1,
      buffer[base + 1] ?? -1,
      buffer[base + 2] ?? -1,
      buffer[base + 3] ?? -1,
      buffer[base + 4] ?? -1,
      buffer[base + 5] ?? -1,
      buffer[base + 6] ?? -1,
      buffer[base + 7] ?? -1,
    ]);
  }
  return out;
}

function unlockedHand(ownHand: Uint8Array, locked: Uint8Array): Uint8Array {
  const out = new Uint8Array(TILE_KIND_COUNT());
  for (let i = 0; i < TILE_KIND_COUNT(); i += 1) {
    out[i] = Math.max(0, (ownHand[i] ?? 0) - (locked[i] ?? 0));
  }
  return out;
}

export interface BuildUiSnapshotOptions {
  viewer: number;
  engineRulesVersion: number;
  buffers?: ObservationBuffers;
  /** When false, skip copying full event history (step events still load). */
  includeEventHistory?: boolean;
}

/**
 * Build a viewer-relative UI snapshot from a low-level WASM Game.
 *
 * This is the client-shaped contract. The WASM layer stays Python-symmetric;
 * this function owns seat rotation, unlocked-hand derivation, and event packing.
 */
export function buildUiSnapshot(
  game: GameLike,
  options: BuildUiSnapshotOptions,
): UiSnapshot {
  const viewer = options.viewer;
  if (viewer < 0 || viewer >= PLAYER_COUNT()) {
    throw new RangeError(`viewer seat must be in 0..3, got ${viewer}`);
  }

  const buffers = options.buffers ?? createObservationBuffers();
  const includeHistory = options.includeEventHistory !== false;

  game.observeInto(viewer, buffers.tileObs, buffers.melds, buffers.river, buffers.meta);

  const historyCount = includeHistory ? game.eventsInto(viewer, buffers.events) : 0;
  const stepCount = game.stepEventsInto(viewer, buffers.stepEvents);

  const meta = buffers.meta;
  const ownHand = copyPlane(buffers.tileObs, 0);
  const exchangeSelection = copyPlane(buffers.tileObs, 1);
  const lockedTiles = [
    copyPlane(buffers.tileObs, 2),
    copyPlane(buffers.tileObs, 3),
    copyPlane(buffers.tileObs, 4),
    copyPlane(buffers.tileObs, 5),
  ] as const;
  const discardCounts = [
    copyPlane(buffers.tileObs, 6),
    copyPlane(buffers.tileObs, 7),
    copyPlane(buffers.tileObs, 8),
    copyPlane(buffers.tileObs, 9),
  ] as const;

  const handCounts = [
    meta[24] ?? 0,
    meta[25] ?? 0,
    meta[26] ?? 0,
    meta[27] ?? 0,
  ] as const;
  const unlocked = unlockedHand(ownHand, lockedTiles[0]);
  const unlockedHandCounts = [
    Math.max(0, handCounts[0] - lockedTiles[0].reduce((a, b) => a + b, 0)),
    Math.max(0, handCounts[1] - lockedTiles[1].reduce((a, b) => a + b, 0)),
    Math.max(0, handCounts[2] - lockedTiles[2].reduce((a, b) => a + b, 0)),
    Math.max(0, handCounts[3] - lockedTiles[3].reduce((a, b) => a + b, 0)),
  ] as const;

  const [low, high] = readMaskWords(game.legalActionMask);
  const decision = game.decision;
  const finished = game.phase === PHASE_FINISHED();

  const absoluteScores = Array.from(game.scores());
  const absoluteMissing = Array.from(game.missingSuits());
  const absoluteMultipliers = Array.from(game.maxWinMultipliers());

  return {
    schemaVersion: UI_SNAPSHOT_SCHEMA_VERSION,
    engineRulesVersion: options.engineRulesVersion,
    phase: game.phase,
    decisionActor:
      decision == null || decision.length < 1
        ? null
        : relativeSeat(Number(decision[0] ?? 0), viewer),
    dealer: relativeSeat(game.dealer, viewer),
    exchangeDirection: game.exchangeDirection,
    wallRemaining: game.wallRemaining,
    scores: relativeQuad(absoluteScores, viewer),
    missingSuits: relativeQuad(absoluteMissing, viewer),
    ownHand,
    unlockedHand: unlocked,
    exchangeSelection,
    lockedTiles,
    discardCounts,
    melds: parseMelds(buffers.melds),
    river: parseRiver(buffers.river, meta[9] ?? 0),
    handCounts,
    unlockedHandCounts,
    pendingSource: meta[7] ?? -1,
    pendingTile: meta[8] ?? -1,
    pendingResponseFlags: meta[29] ?? 0,
    drawTile: meta[5] ?? -1,
    replacementDraw: (meta[6] ?? 0) !== 0,
    hasWon: [
      (meta[20] ?? 0) !== 0,
      (meta[21] ?? 0) !== 0,
      (meta[22] ?? 0) !== 0,
      (meta[23] ?? 0) !== 0,
    ],
    maxWinMultipliers: relativeQuad(absoluteMultipliers, viewer),
    legalActionMask: [low, high],
    legalActionIds: legalActionIdsFromMask(low, high),
    eventHistory: parseEvents(buffers.events, historyCount),
    stepEvents: parseEvents(buffers.stepEvents, stepCount),
    terminationReason: finished ? (game.terminationReason ?? null) : null,
    rankings: finished
      ? (Array.from(game.rankings()).map((seat) =>
          relativeSeat(Number(seat), viewer),
        ) as [number, number, number, number])
      : null,
  };
}
