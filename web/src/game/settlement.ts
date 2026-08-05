import type { EventRecord } from "../../../engine/wasm/js/src/types";
import { EventKind } from "./domain";

/** Return the event indexes of settlement stages that actually moved points. */
export function scoreChangingSettlementStageIndexes(
  events: readonly EventRecord[],
): Set<number> {
  const indexes = new Set<number>();
  let activeStage = -1;
  for (let index = 0; index < events.length; index += 1) {
    const event = events[index]!;
    if (event[0] === EventKind.SettlementStage) {
      activeStage = index;
      continue;
    }
    if (event[0] === EventKind.GameEnd) {
      activeStage = -1;
      continue;
    }
    if (activeStage >= 0 && event[0] === EventKind.Payment && event[5] > 0) {
      indexes.add(activeStage);
    }
  }
  return indexes;
}

export function settlementStageLabel(events: readonly EventRecord[]): string {
  const stages = scoreChangingSettlementStageIndexes(events);
  const latestIndex = [...stages].at(-1);
  if (latestIndex == null) return "终局";
  return events[latestIndex]![4] === 0 ? "查花猪" : "查大叫";
}
