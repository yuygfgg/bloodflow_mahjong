import { describe, expect, it } from "vitest";
import type { UiSnapshot } from "../../../engine/wasm/js/src/types";
import {
  Action,
  Phase,
  actionButtons,
  expandHistogram,
  replayFromIdentifier,
  replayIdentifier,
  tileLabel,
  windForSeat,
} from "./domain";

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
});
