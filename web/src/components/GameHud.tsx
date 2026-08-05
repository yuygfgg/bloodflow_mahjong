import { Check, Lightbulb } from "lucide-react";
import type { BotProfile } from "../../../engine/wasm/js/src/protocol";
import type { UiSnapshot } from "../../../engine/wasm/js/src/types";
import type { GameConfig } from "../game/useGameEngine";
import {
  Action,
  Phase,
  type ExchangeSelection,
  SEAT_NAMES,
  SUIT_LABELS,
  actionButtons,
  discardTileAction,
  exchangeDirectionLabel,
  exchangeTileAction,
  expandHistogram,
  isActionLegal,
  phaseTitle,
  tileLabel,
  tileSuit,
  windForSeat,
} from "../game/domain";

interface SeatHudProps {
  snapshot: UiSnapshot;
  config: GameConfig;
  pendingMissingSuit: number | null;
}

const PROFILE_LABEL: Record<BotProfile, string> = {
  "rule-fast": "极速",
  "rule-ev": "估值",
  "rule-nn": "神经网络",
};

export function SeatHuds({
  snapshot,
  config,
  pendingMissingSuit,
}: SeatHudProps) {
  return (
    <div className="seat-huds" aria-label="玩家信息">
      {[0, 1, 2, 3].map((seat) => {
        const absolute = (config.humanSeat + seat) & 3;
        const profile = config.botProfiles[absolute]!;
        const missing =
          seat === 0 && pendingMissingSuit != null
            ? pendingMissingSuit
            : (snapshot.missingSuits[seat] ?? -1);
        return (
          <section
            className={`seat-hud seat-${seat} ${snapshot.decisionActor === seat ? "active" : ""}`}
            key={seat}
            aria-label={`${SEAT_NAMES[seat]} ${snapshot.scores[seat]}分`}
          >
            <div className="seat-wind">
              {windForSeat(seat, snapshot.dealer)}
            </div>
            <div className="seat-copy">
              <strong>{seat === 0 ? "玩家" : PROFILE_LABEL[profile]}</strong>
              <span>{snapshot.scores[seat].toLocaleString("zh-CN")}</span>
            </div>
            {snapshot.dealer === seat && (
              <span className="dealer-mark">庄</span>
            )}
            {missing >= 0 && (
              <span
                className={`missing-mark suit-${missing} ${seat === 0 && pendingMissingSuit != null ? "pending" : ""}`}
              >
                {SUIT_LABELS[missing]}
              </span>
            )}
            {(snapshot.winCounts[seat] ?? 0) > 0 && (
              <span className="win-count">
                已和 {snapshot.winCounts[seat]} 次
              </span>
            )}
          </section>
        );
      })}
    </div>
  );
}

export function CenterStatus({ snapshot }: { snapshot: UiSnapshot }) {
  return (
    <section className="center-status center-status-a11y" aria-live="polite">
      <strong>余 {snapshot.wallRemaining}</strong>
      <span>{phaseTitle(snapshot)}</span>
      {snapshot.phase === Phase.Exchange && (
        <span>方向 {exchangeDirectionLabel(snapshot.exchangeDirection)}</span>
      )}
      {snapshot.pendingSource >= 0 && (
        <span>来源 {SEAT_NAMES[snapshot.pendingSource]}</span>
      )}
      {snapshot.river.length > 0 && (
        <span>
          最新弃牌 {tileLabel(snapshot.river[snapshot.river.length - 1]!.tile)}
        </span>
      )}
    </section>
  );
}

function ActionTile({
  tile,
  large = false,
}: {
  tile: number;
  large?: boolean;
}) {
  const column = tile % 8;
  const row = Math.floor(tile / 8);
  return (
    <span
      className={`action-tile ${large ? "action-tile-large" : ""}`}
      style={{
        backgroundPosition: `${(column / 7) * 100}% ${(row / 3) * 100}%`,
      }}
      aria-hidden="true"
    />
  );
}

interface ActionBarProps {
  snapshot: UiSnapshot;
  busy: boolean;
  pendingExchangeSelections: readonly ExchangeSelection[];
  pendingMissingSuit: number | null;
  hintPolicy: BotProfile;
  onExchangeTile: (tile: number, sourceKey: string) => void;
  onConfirmExchange: () => void;
  onPendingMissing: (suit: number | null) => void;
  onSubmit: (actionId: number) => void;
  onHintPolicyChange: (policy: BotProfile) => void;
  onHint: (policy: BotProfile) => void;
}

export function ActionBar({
  snapshot,
  busy,
  pendingExchangeSelections,
  pendingMissingSuit,
  hintPolicy,
  onExchangeTile,
  onConfirmExchange,
  onPendingMissing,
  onSubmit,
  onHintPolicyChange,
  onHint,
}: ActionBarProps) {
  const selectedTiles = expandHistogram(snapshot.exchangeSelection);
  selectedTiles.push(
    ...pendingExchangeSelections.map((selection) => selection.tile),
  );
  const responseTile =
    snapshot.pendingTile >= 0 &&
    (snapshot.phase === Phase.HuResponse ||
      snapshot.phase === Phase.MeldResponse)
      ? snapshot.pendingTile
      : null;

  return (
    <div className="action-region">
      {snapshot.phase === Phase.Exchange && snapshot.decisionActor === 0 && (
        <div className="phase-panel exchange-panel">
          <div className="phase-banner">
            <strong>换三张</strong>
            <span>
              向{exchangeDirectionLabel(snapshot.exchangeDirection)}交换
            </span>
            <span>已选 {selectedTiles.length}/3</span>
          </div>
          <div className="selection-tray" aria-label="已选交换牌">
            {selectedTiles.map((tile, index) => (
              <span key={`${tile}-${index}`}>{tileLabel(tile)}</span>
            ))}
          </div>
          {selectedTiles.length === 3 &&
            pendingExchangeSelections.length > 0 && (
              <button
                className="menu-button action-confirm"
                type="button"
                disabled={busy}
                onClick={onConfirmExchange}
              >
                <Check aria-hidden="true" />
                确认换牌
              </button>
            )}
        </div>
      )}

      {snapshot.phase === Phase.ChooseMissing &&
        snapshot.decisionActor === 0 && (
          <div className="phase-panel missing-panel">
            <div className="phase-banner">
              <strong>定缺</strong>
              <span>选择本局定缺花色</span>
            </div>
            <div className="suit-actions" role="group" aria-label="定缺花色">
              {SUIT_LABELS.map((label, suit) => {
                const action = Action.ChooseMissingOffset + suit;
                return (
                  <button
                    className={`menu-button suit-button ${pendingMissingSuit === suit ? "selected" : ""}`}
                    key={label}
                    type="button"
                    disabled={busy || !isActionLegal(snapshot, action)}
                    onClick={() => onPendingMissing(suit)}
                  >
                    {label}
                  </button>
                );
              })}
              <button
                className="menu-button action-confirm"
                type="button"
                disabled={busy || pendingMissingSuit == null}
                onClick={() => {
                  if (pendingMissingSuit != null) {
                    onSubmit(Action.ChooseMissingOffset + pendingMissingSuit);
                  }
                }}
              >
                <Check aria-hidden="true" />
                确认定缺
              </button>
            </div>
          </div>
        )}

      {snapshot.decisionActor === 0 &&
        snapshot.phase !== Phase.Exchange &&
        snapshot.phase !== Phase.ChooseMissing &&
        snapshot.phase !== Phase.Finished && (
          <div className="action-bar" role="toolbar" aria-label="可用动作">
            {responseTile != null && (
              <div
                className="response-tile"
                aria-label={`响应牌 ${tileLabel(responseTile)}`}
              >
                <ActionTile tile={responseTile} large />
                <span>{tileLabel(responseTile)}</span>
              </div>
            )}
            {actionButtons(snapshot).map((action) => (
              <button
                className={`game-action tone-${action.tone} ${action.tile != null && responseTile == null ? "has-tile" : ""}`}
                key={action.actionId}
                type="button"
                disabled={busy}
                onClick={() => onSubmit(action.actionId)}
              >
                {action.tile != null && responseTile == null && (
                  <ActionTile tile={action.tile} />
                )}
                {action.label}
              </button>
            ))}
            <select
              className="hint-policy"
              aria-label="提示 AI"
              title="提示 AI"
              value={hintPolicy}
              disabled={busy}
              onChange={(event) =>
                onHintPolicyChange(event.target.value as BotProfile)
              }
            >
              <option value="rule-fast">极速规则</option>
              <option value="rule-ev">估值搜索</option>
              <option value="rule-nn">神经网络</option>
            </select>
            <button
              className="game-action icon-action"
              type="button"
              disabled={busy}
              title="提示"
              aria-label={`使用${PROFILE_LABEL[hintPolicy]}提示`}
              onClick={() => onHint(hintPolicy)}
            >
              <Lightbulb aria-hidden="true" />
            </button>
          </div>
        )}

      <AccessibleTileControls
        snapshot={snapshot}
        busy={busy}
        pendingExchangeSelections={pendingExchangeSelections}
        onExchangeTile={onExchangeTile}
        onSubmit={onSubmit}
      />
    </div>
  );
}

interface AccessibleTileControlsProps {
  snapshot: UiSnapshot;
  busy: boolean;
  pendingExchangeSelections: readonly ExchangeSelection[];
  onExchangeTile: (tile: number, sourceKey: string) => void;
  onSubmit: (actionId: number) => void;
}

function AccessibleTileControls({
  snapshot,
  busy,
  pendingExchangeSelections,
  onExchangeTile,
  onSubmit,
}: AccessibleTileControlsProps) {
  const hand = expandHistogram(snapshot.unlockedHand);
  const phase = snapshot.phase;
  if (
    snapshot.decisionActor !== 0 ||
    (phase !== Phase.Exchange && phase !== Phase.Turn)
  ) {
    return null;
  }
  const selectedKeys = new Set(
    pendingExchangeSelections.map((selection) => selection.sourceKey),
  );
  const exchangeFull =
    snapshot.exchangeSelectedCount + pendingExchangeSelections.length >= 3;
  return (
    <div className="accessible-hand" aria-label="手牌操作">
      {hand.map((tile, index) => {
        const action =
          phase === Phase.Exchange
            ? exchangeTileAction(tile)
            : discardTileAction(tile);
        const legal = isActionLegal(snapshot, action);
        const sourceKey = `accessible-hand:${index}`;
        const selected =
          phase === Phase.Exchange && selectedKeys.has(sourceKey);
        const disabled =
          busy ||
          !legal ||
          (phase === Phase.Exchange && exchangeFull && !selected);
        return (
          <button
            type="button"
            key={`${tile}-${index}`}
            disabled={disabled}
            aria-pressed={selected}
            aria-label={`${phase === Phase.Exchange ? "选择交换" : "打出"}${tileLabel(tile)}`}
            onClick={() => {
              if (phase === Phase.Exchange) {
                onExchangeTile(tile, sourceKey);
              } else {
                onSubmit(action);
              }
            }}
          >
            {tileLabel(tile)}
          </button>
        );
      })}
    </div>
  );
}

export function canSelectExchangeTile(
  snapshot: UiSnapshot,
  tile: number,
): boolean {
  const action = exchangeTileAction(tile);
  if (!isActionLegal(snapshot, action)) return false;
  const suit = snapshot.exchangeSelectionSuit;
  return suit < 0 || tileSuit(tile) === suit;
}
