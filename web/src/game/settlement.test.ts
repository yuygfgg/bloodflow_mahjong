import { describe, expect, it } from "vitest";
import type { EventRecord } from "../../../engine/wasm/js/src/types";
import { EventKind } from "./domain";
import {
  scoreChangingSettlementStageIndexes,
  settlementStageLabel,
} from "./settlement";

function event(kind: number, flags = 0, value = 0): EventRecord {
  return [kind, -1, -1, -1, flags, value, 0, 0];
}

describe("settlement display stages", () => {
  it("marks only stages followed by an actual payment", () => {
    const events = [
      event(EventKind.SettlementStage, 0),
      event(EventKind.Payment, 0, 100),
      event(EventKind.SettlementStage, 1),
      event(EventKind.GameEnd),
    ];

    expect([...scoreChangingSettlementStageIndexes(events)]).toEqual([0]);
  });

  it("does not treat a zero payment as a score change", () => {
    const events = [
      event(EventKind.SettlementStage, 0),
      event(EventKind.Payment, 0, 0),
      event(EventKind.SettlementStage, 1),
      event(EventKind.Payment, 0, 200),
      event(EventKind.GameEnd),
    ];

    expect([...scoreChangingSettlementStageIndexes(events)]).toEqual([2]);
  });

  it("keeps both stages when both move points", () => {
    const events = [
      event(EventKind.SettlementStage, 0),
      event(EventKind.Payment, 0, 100),
      event(EventKind.SettlementStage, 1),
      event(EventKind.Payment, 0, 200),
      event(EventKind.GameEnd),
    ];

    expect([...scoreChangingSettlementStageIndexes(events)]).toEqual([0, 2]);
    expect(settlementStageLabel(events)).toBe("查大叫");
  });

  it("uses the terminal label when settlement moves no points", () => {
    const events = [
      event(EventKind.SettlementStage, 0),
      event(EventKind.SettlementStage, 1),
      event(EventKind.GameEnd),
    ];

    expect([...scoreChangingSettlementStageIndexes(events)]).toEqual([]);
    expect(settlementStageLabel(events)).toBe("终局");
  });
});
