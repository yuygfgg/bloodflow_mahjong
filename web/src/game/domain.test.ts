import { describe, expect, it } from "vitest";
import type {
  EventRecord,
  RiverEntry,
  UiSnapshot,
} from "../../../engine/wasm/js/src/types";
import {
  Action,
  EVENT_FLAG_ROB_KONG,
  EVENT_FLAG_SELF_DRAW,
  EventKind,
  MeldKind,
  Phase,
  actionButtons,
  claimedRiverIndexes,
  expandHistogram,
  replayFromIdentifier,
  replayIdentifier,
  tileLabel,
  windForSeat,
} from "./domain";

function event(
  kind: number,
  actor: number,
  target: number,
  tile: number,
  flags = 0,
): EventRecord {
  return [kind, actor, target, tile, flags, 0, 0, -1];
}

function snapshotWith(
  actions: number[],
  phase: number = Phase.Turn,
): UiSnapshot {
  return { phase, legalActionIds: actions } as unknown as UiSnapshot;
}

describe("game presentation mappings", () => {
  it("uses the engine tile ordering", () => {
    expect(tileLabel(0)).toBe("1万");
    expect(tileLabel(17)).toBe("9条");
    expect(tileLabel(26)).toBe("9筒");
  });

  it("rotates winds around a relative dealer", () => {
    expect([0, 1, 2, 3].map((seat) => windForSeat(seat, 2))).toEqual([
      "西",
      "北",
      "东",
      "南",
    ]);
  });

  it("expands tile histograms in stable sorted order", () => {
    const histogram = new Uint8Array(27);
    histogram[0] = 2;
    histogram[11] = 1;
    expect(expandHistogram(histogram)).toEqual([0, 0, 11]);
  });

  it("orders response actions as hu, kong, pong, pass", () => {
    expect(
      actionButtons(
        snapshotWith(
          [Action.Pass, Action.Pong, Action.ExposedKong, Action.Hu],
          Phase.HuResponse,
        ),
      ).map((button) => button.label),
    ).toEqual(["胡", "直杠", "碰", "过"]);
  });

  it("attaches the pending tile to response actions", () => {
    const buttons = actionButtons({
      ...snapshotWith(
        [Action.Pass, Action.Pong, Action.ExposedKong, Action.Hu],
        Phase.HuResponse,
      ),
      pendingTile: 8,
    } as UiSnapshot);
    expect(buttons.filter((button) => button.tile != null)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: "胡", tile: 8 }),
        expect.objectContaining({ label: "直杠", tile: 8 }),
        expect.objectContaining({ label: "碰", tile: 8 }),
      ]),
    );
  });

  it("attaches a self-draw tile to the hu action", () => {
    const buttons = actionButtons({
      ...snapshotWith([Action.Hu], Phase.Turn),
      drawTile: 17,
      pendingTile: -1,
    } as UiSnapshot);
    expect(buttons).toEqual([
      expect.objectContaining({ label: "自摸", tile: 17 }),
    ]);
  });

  it("round-trips replay identifiers", () => {
    const json = JSON.stringify({ seed: "42", actions: [0, 1, 2] });
    expect(replayFromIdentifier(replayIdentifier(json))).toBe(json);
  });

  it("marks discards claimed by pong and exposed kong", () => {
    const river: RiverEntry[] = [
      { ownerRelative: 1, tile: 4 },
      { ownerRelative: 2, tile: 8 },
    ];
    const events = [
      event(EventKind.Discard, 1, -1, 4),
      event(EventKind.Meld, 0, 1, 4, MeldKind.Pong),
      event(EventKind.Discard, 2, -1, 8),
      event(EventKind.Meld, 3, 2, 8, MeldKind.ExposedKong),
    ];

    expect([...claimedRiverIndexes(river, events)]).toEqual([0, 1]);
  });

  it("marks one physical discard for multiple discard wins", () => {
    const river: RiverEntry[] = [{ ownerRelative: 2, tile: 12 }];
    const events = [
      event(EventKind.Discard, 2, -1, 12),
      event(EventKind.Hu, 3, 2, 12),
      event(EventKind.Hu, 0, 2, 12),
    ];

    expect([...claimedRiverIndexes(river, events)]).toEqual([0]);
  });

  it("does not claim a river tile for self draw or robbed kong", () => {
    const river: RiverEntry[] = [{ ownerRelative: 1, tile: 6 }];
    const events = [
      event(EventKind.Discard, 1, -1, 6),
      event(EventKind.Hu, 0, -1, 6, EVENT_FLAG_SELF_DRAW),
      event(EventKind.Hu, 2, 1, 6, EVENT_FLAG_ROB_KONG),
      event(EventKind.Meld, 1, -1, 6, MeldKind.AddedKong),
      event(EventKind.Meld, 3, -1, 6, MeldKind.ConcealedKong),
    ];

    expect([...claimedRiverIndexes(river, events)]).toEqual([]);
  });

  it("claims the correct repeated discard and aligns retained event history", () => {
    const river: RiverEntry[] = [
      { ownerRelative: 1, tile: 3 },
      { ownerRelative: 2, tile: 9 },
      { ownerRelative: 1, tile: 3 },
    ];
    const events = [
      event(EventKind.Discard, 1, -1, 3),
      event(EventKind.Hu, 0, 1, 3),
    ];

    expect([...claimedRiverIndexes(river, events)]).toEqual([2]);
  });
});
