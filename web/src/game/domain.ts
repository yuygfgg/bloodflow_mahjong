import type { UiSnapshot } from "../../../engine/wasm/js/src/types";

export const TILE_KIND_COUNT = 27;

export const Phase = {
  Exchange: 0,
  ChooseMissing: 1,
  Turn: 2,
  HuResponse: 3,
  MeldResponse: 4,
  Finished: 5,
} as const;

export const Action = {
  ExchangeOffset: 0,
  ChooseMissingOffset: 27,
  DiscardOffset: 30,
  Hu: 57,
  Pong: 58,
  ExposedKong: 59,
  ConcealedKongOffset: 60,
  AddedKongOffset: 87,
  Pass: 114,
} as const;

export const EventKind = {
  Action: 0,
  GameStart: 1,
  TurnStart: 2,
  Draw: 3,
  Discard: 4,
  ExchangeComplete: 5,
  MissingRevealed: 6,
  Meld: 7,
  Hu: 8,
  Payment: 9,
  GameEnd: 10,
  SettlementStage: 11,
} as const;

export const MeldKind = {
  Pong: 0,
  ExposedKong: 1,
  AddedKong: 2,
  ConcealedKong: 3,
} as const;

export const EVENT_FLAG_REPLACEMENT_DRAW = 1 << 0;
export const EVENT_FLAG_LAST_WALL_TILE = 1 << 1;
export const EVENT_FLAG_AFTER_KONG = 1 << 2;
export const EVENT_FLAG_SELF_DRAW = 1 << 4;
export const EVENT_FLAG_ROB_KONG = 1 << 5;
export const EVENT_FLAG_HEAVENLY = 1 << 6;
export const EVENT_FLAG_EARTHLY = 1 << 7;

export const SUIT_LABELS = ["万", "条", "筒"] as const;
export const WIND_LABELS = ["东", "南", "西", "北"] as const;
export const SEAT_NAMES = ["玩家", "下家", "对家", "上家"] as const;

export interface ExchangeSelection {
  tile: number;
  sourceKey: string;
}

export const PATTERN_LABELS = [
  "平胡",
  "断幺九",
  "碰碰胡",
  "清碰碰胡",
  "七对",
  "清七对",
  "将七对",
  "龙七对",
  "清龙七对",
  "双龙七对",
  "三龙七对",
  "将双龙七对",
  "将三龙七对",
  "十八罗汉",
  "清十八罗汉",
  "幺九",
  "清幺九",
  "清一色",
  "金钩钓",
  "清金钩钓",
  "抢杠胡",
  "杠上炮",
  "杠上开花",
  "海底捞月",
  "天胡",
  "地胡",
] as const;

export function tileSuit(tile: number): number {
  return Math.floor(tile / 9);
}

export function tileLabel(tile: number): string {
  if (tile < 0 || tile >= TILE_KIND_COUNT) {
    return "暗牌";
  }
  return `${(tile % 9) + 1}${SUIT_LABELS[tileSuit(tile)]}`;
}

export function windForSeat(
  relativeSeat: number,
  dealerRelative: number,
): string {
  return WIND_LABELS[(relativeSeat - dealerRelative + 4) & 3]!;
}

export function expandHistogram(histogram: ArrayLike<number>): number[] {
  const tiles: number[] = [];
  for (let tile = 0; tile < TILE_KIND_COUNT; tile += 1) {
    for (let copy = 0; copy < Number(histogram[tile] ?? 0); copy += 1) {
      tiles.push(tile);
    }
  }
  return tiles;
}

export function legalDiscardTiles(snapshot: UiSnapshot): Set<number> {
  return new Set(
    snapshot.legalActionIds
      .filter((action) => action >= Action.DiscardOffset && action < Action.Hu)
      .map((action) => action - Action.DiscardOffset),
  );
}

export function exchangeTileAction(tile: number): number {
  return Action.ExchangeOffset + tile;
}

export function missingSuitAction(suit: number): number {
  return Action.ChooseMissingOffset + suit;
}

export function discardTileAction(tile: number): number {
  return Action.DiscardOffset + tile;
}

export function isActionLegal(snapshot: UiSnapshot, action: number): boolean {
  return snapshot.legalActionIds.includes(action);
}

export interface ActionButtonSpec {
  actionId: number;
  label: string;
  tone: "hu" | "kong" | "pong" | "pass" | "neutral";
  /** Tile displayed with an action when the action has a concrete tile. */
  tile?: number;
}

export function actionButtons(snapshot: UiSnapshot): ActionButtonSpec[] {
  const result: ActionButtonSpec[] = [];
  const legal = snapshot.legalActionIds;
  if (legal.includes(Action.Hu)) {
    result.push({
      actionId: Action.Hu,
      label: snapshot.phase === Phase.Turn ? "自摸" : "胡",
      tone: "hu",
      tile:
        snapshot.phase === Phase.Turn && snapshot.drawTile >= 0
          ? snapshot.drawTile
          : snapshot.pendingTile >= 0
            ? snapshot.pendingTile
            : undefined,
    });
  }
  if (legal.includes(Action.ExposedKong)) {
    result.push({
      actionId: Action.ExposedKong,
      label: "直杠",
      tone: "kong",
      tile: snapshot.pendingTile >= 0 ? snapshot.pendingTile : undefined,
    });
  }
  for (const actionId of legal) {
    if (
      actionId >= Action.ConcealedKongOffset &&
      actionId < Action.AddedKongOffset
    ) {
      result.push({
        actionId,
        label: `暗杠 ${tileLabel(actionId - Action.ConcealedKongOffset)}`,
        tone: "kong",
        tile: actionId - Action.ConcealedKongOffset,
      });
    }
  }
  for (const actionId of legal) {
    if (actionId >= Action.AddedKongOffset && actionId < Action.Pass) {
      result.push({
        actionId,
        label: `碰杠 ${tileLabel(actionId - Action.AddedKongOffset)}`,
        tone: "kong",
        tile: actionId - Action.AddedKongOffset,
      });
    }
  }
  if (legal.includes(Action.Pong)) {
    result.push({
      actionId: Action.Pong,
      label: "碰",
      tone: "pong",
      tile: snapshot.pendingTile >= 0 ? snapshot.pendingTile : undefined,
    });
  }
  if (legal.includes(Action.Pass)) {
    result.push({ actionId: Action.Pass, label: "过", tone: "pass" });
  }
  return result;
}

export function phaseTitle(snapshot: UiSnapshot): string {
  switch (snapshot.phase) {
    case Phase.Exchange:
      return "换三张";
    case Phase.ChooseMissing:
      return "定缺";
    case Phase.Turn:
      if (snapshot.replacementDraw) return "杠后补牌";
      if (snapshot.wallRemaining === 0 && snapshot.drawTile >= 0) return "海底";
      return snapshot.decisionActor === 0 ? "请出牌" : "等待对手";
    case Phase.HuResponse:
      return snapshot.pendingResponseFlags & 1
        ? "等待抢杠胡"
        : `响应 ${tileLabel(snapshot.pendingTile)}`;
    case Phase.MeldResponse:
      return `响应 ${tileLabel(snapshot.pendingTile)}`;
    case Phase.Finished:
      return "本局结束";
    default:
      return "血流成河";
  }
}

export function exchangeDirectionLabel(direction: number): string {
  return ["", "左", "对家", "右"][direction] ?? "";
}

export function patternLabels(bits: number): string[] {
  return PATTERN_LABELS.filter((_, index) => (bits & (1 << index)) !== 0);
}

export function replayIdentifier(json: string): string {
  const bytes = new TextEncoder().encode(json);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary)
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replace(/=+$/, "");
}

export function replayFromIdentifier(identifier: string): string {
  const normalized = identifier
    .trim()
    .replaceAll("-", "+")
    .replaceAll("_", "/");
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
  const binary = atob(padded);
  return new TextDecoder().decode(
    Uint8Array.from(binary, (value) => value.charCodeAt(0)),
  );
}
