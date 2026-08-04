/**
 * Client-facing types built on top of the low-level WASM Game API.
 */

/** One fixed-width event record: [kind, actor, target, tile, flags, value, aux, reserved]. */
export type EventRecord = readonly [
  kind: number,
  actorRelative: number,
  targetRelative: number,
  tile: number,
  flags: number,
  value: number,
  aux: number,
  reserved: number,
];

export interface MeldView {
  tile: number;
  kind: number;
  sourceRelative: number;
}

export interface RiverEntry {
  tile: number;
  ownerRelative: number;
}

/**
 * Viewer-relative UI snapshot.
 *
 * Seat index 0 is always the viewer. Opponents are 1 (right), 2 (across),
 * 3 (left) in table order. Hidden information stays filtered by the engine.
 */
export interface UiSnapshot {
  schemaVersion: number;
  engineRulesVersion: number;
  phase: number;
  /** Relative decision actor, or null when the game is finished. */
  decisionActor: number | null;
  /** Relative dealer seat. */
  dealer: number;
  exchangeDirection: number;
  wallRemaining: number;
  /** Relative seat scores. */
  scores: readonly [number, number, number, number];
  /** Relative missing suits; -1 means unset. */
  missingSuits: readonly [number, number, number, number];
  /** Viewer concealed histogram including locked winning structures. */
  ownHand: Uint8Array;
  /** Viewer playable tiles after subtracting locked counts. */
  unlockedHand: Uint8Array;
  /** Viewer exchange selection histogram. */
  exchangeSelection: Uint8Array;
  /** Locked winning-tile histograms for relative seats 0..3. */
  lockedTiles: readonly [Uint8Array, Uint8Array, Uint8Array, Uint8Array];
  /** Per-relative-seat discard histograms. */
  discardCounts: readonly [Uint8Array, Uint8Array, Uint8Array, Uint8Array];
  /** Relative-seat melds. */
  melds: readonly [
    readonly MeldView[],
    readonly MeldView[],
    readonly MeldView[],
    readonly MeldView[],
  ];
  /** Chronological river with relative owners. */
  river: readonly RiverEntry[];
  /** Total concealed counts for relative seats. */
  handCounts: readonly [number, number, number, number];
  /** Unlocked concealed counts for relative seats. */
  unlockedHandCounts: readonly [number, number, number, number];
  pendingSource: number;
  pendingTile: number;
  pendingResponseFlags: number;
  /** Visible draw tile for the viewer, or -1. */
  drawTile: number;
  replacementDraw: boolean;
  hasWon: readonly [boolean, boolean, boolean, boolean];
  maxWinMultipliers: readonly [number, number, number, number];
  /** Little-endian legal mask words `[low, high]`. */
  legalActionMask: readonly [bigint, bigint];
  legalActionIds: readonly number[];
  /** Full retained history visible to the viewer. */
  eventHistory: readonly EventRecord[];
  /** Events from the most recent step only. */
  stepEvents: readonly EventRecord[];
  terminationReason: number | null;
  /** Relative rankings when finished; otherwise null. */
  rankings: readonly [number, number, number, number] | null;
}

export interface ObservationBuffers {
  tileObs: Uint8Array;
  melds: Uint8Array;
  river: Uint8Array;
  meta: Int32Array;
  events: Int32Array;
  stepEvents: Int32Array;
}

/** Narrow structural view of the WASM Game class used by the snapshot builder. */
export interface GameLike {
  readonly phase: number;
  readonly decision: ArrayLike<number> | undefined;
  readonly legalActionMask: BigUint64Array | ArrayLike<bigint | number>;
  readonly dealer: number;
  readonly exchangeDirection: number;
  readonly wallRemaining: number;
  readonly terminationReason: number | undefined;
  scores(): ArrayLike<number>;
  missingSuits(): ArrayLike<number>;
  rankings(): ArrayLike<number>;
  maxWinMultipliers(): ArrayLike<number>;
  observeInto(
    viewer: number,
    tileObs: Uint8Array,
    melds: Uint8Array,
    river: Uint8Array,
    meta: Int32Array,
  ): void;
  eventsInto(viewer: number, output: Int32Array): number;
  stepEventsInto(viewer: number, output: Int32Array): number;
}
