from __future__ import annotations

import random
from typing import Callable

import bloodflow_mahjong as bm
import numpy as np

from .policy import make_policy
from .tiles import (
    DIRECTION_NAMES,
    MELD_NAMES,
    SEAT_NAMES,
    SUIT_NAMES,
    kind_text,
)


def format_event(event) -> str | None:
    """把一条 8 宽事件记录格式化为玩家(viewer=0)视角的中文日志文本。"""
    kind = int(event[0])
    actor = int(event[1])
    target = int(event[2])
    tile = int(event[3])
    flags = int(event[4])
    value = int(event[5])
    name = SEAT_NAMES[actor] if 0 <= actor < 4 else "?"
    if kind == bm.EventKind.ACTION:
        if bm.ACTION_ADDED_KONG_OFFSET <= value < bm.ACTION_PASS:
            declared_tile = tile if tile >= 0 else value - bm.ACTION_ADDED_KONG_OFFSET
            return f"{name}声明碰杠 {kind_text(declared_tile)}"
        return None
    if kind == bm.EventKind.GAME_START:
        return f"牌局开始,换牌方向: {DIRECTION_NAMES.get(flags, '?')}"
    if kind == bm.EventKind.TURN_START:
        return "庄家开始行牌"
    if kind == bm.EventKind.DRAW:
        if actor == 0 and tile >= 0:
            extra = ""
            if flags & bm.EventFlag.REPLACEMENT_DRAW:
                extra = " (杠后补牌)"
            elif flags & bm.EventFlag.LAST_WALL_TILE:
                extra = " (海底)"
            return f"你摸到 {kind_text(tile)}{extra}"
        return f"{name}摸牌"
    if kind == bm.EventKind.DISCARD:
        extra = ""
        if flags & bm.EventFlag.AFTER_KONG:
            extra = " (杠后)"
        elif flags & (1 << 4):
            extra = " (摸切)"
        elif flags & (1 << 5):
            extra = " (碰后)"
        return f"{name}打出 {kind_text(tile)}{extra}"
    if kind == bm.EventKind.EXCHANGE_COMPLETE:
        return "换三张完成"
    if kind == bm.EventKind.MISSING_REVEALED:
        return f"{name}定缺: {SUIT_NAMES[value]}"
    if kind == bm.EventKind.MELD:
        return f"{name}{MELD_NAMES[flags]} {kind_text(tile)}"
    if kind == bm.EventKind.HU:
        extra = ""
        if flags & bm.EventFlag.ROB_KONG:
            extra = " (抢杠)"
        elif flags & bm.EventFlag.SELF_DRAW:
            extra = " (自摸)"
        return f"{name}和牌 {kind_text(tile)} ×{value}{extra}"
    if kind == bm.EventKind.PAYMENT:
        return f"{SEAT_NAMES[actor]} 支付 {value} 分给 {SEAT_NAMES[target]}"
    if kind == bm.EventKind.GAME_END:
        if flags & bm.EventFlag.LAST_WALL_TILE:
            return "牌墙摸完,牌局结束"
        return "牌局结束"
    return None


class GameController:
    """封装 bm.Game,驱动「人类 seat 暂停、AI seat 自动」的对局循环。

    回调(可整体替换):
    - on_state_changed(): 状态已推进,调用 view() 刷新
    - on_log(lines: list[str]): 追加日志行
    - on_finished(rankings, scores): 终局
    """

    def __init__(
        self,
        human_seat: int = 0,
        difficulties: dict[int, str] | None = None,
        on_state_changed: Callable[[], None] | None = None,
        on_log: Callable[[list[str]], None] | None = None,
        on_finished: Callable[[tuple, tuple], None] | None = None,
    ) -> None:
        self.human_seat = human_seat
        self.difficulties = dict(difficulties or {})
        self.policies = {
            seat: make_policy(d)
            for seat, d in self.difficulties.items()
            if seat != human_seat
        }
        self.on_state_changed = on_state_changed
        self.on_log = on_log
        self.on_finished = on_finished
        self.human_proxy: Callable[[bm.Game], int | None] | None = None
        self.game = bm.Game()
        self._event_buf = np.empty(
            (bm.EVENT_HISTORY_CAPACITY, bm.EVENT_RECORD_WIDTH), dtype=np.int32
        )
        self._obs_tiles = np.empty((bm.TILE_OBSERVATION_PLANES, 27), dtype=np.uint8)
        self._obs_melds = np.empty((4, 4, 3), dtype=np.uint8)
        self._obs_river = np.empty((bm.RIVER_TILE_CAPACITY, 2), dtype=np.uint8)
        self._obs_meta = np.empty((bm.META_OBSERVATION_WIDTH,), dtype=np.int32)

    # ------------------------------------------------------------------ 驱动

    def start(self, seed: int | None = None) -> None:
        if seed is None:
            seed = random.randrange(1 << 63)
        self.game.reset(seed)
        n = self.game.events_into(self.human_seat, self._event_buf)
        if self.on_log:
            lines = [
                line for i in range(n) if (line := format_event(self._event_buf[i]))
            ]
            if lines:
                self.on_log(lines)
        self._drive()
        self._emit_state()

    def submit(self, action: int) -> bool:
        """提交人类动作;合法则推进并继续驱动 AI,返回是否接受。"""
        if not self.human_must_act():
            return False
        if not self.is_legal(action):
            return False
        self._step(action)
        self._drive()
        self._emit_state()
        return True

    def _step(self, action: int) -> None:
        self.game.step_id(action)
        n = self.game.step_events_into(self.human_seat, self._event_buf)
        if self.on_log and n:
            lines = [
                line for i in range(n) if (line := format_event(self._event_buf[i]))
            ]
            if lines:
                self.on_log(lines)

    def _drive(self) -> None:
        while self.game.phase != bm.PHASE_FINISHED:
            decision = self.game.decision
            if decision is None:
                break
            actor, _phase = decision
            if actor == self.human_seat:
                if self.human_proxy is None:
                    break
                action = self.human_proxy(self.game)
            else:
                action = self.policies[actor](self.game)
            if action is None:
                break
            self._step(action)

    def _emit_state(self) -> None:
        if self.game.phase == bm.PHASE_FINISHED and self.on_finished:
            self.on_finished(self.game.rankings(), self.game.scores())
        if self.on_state_changed:
            self.on_state_changed()

    # ------------------------------------------------------------------ 查询

    def human_must_act(self) -> bool:
        if self.game.phase == bm.PHASE_FINISHED:
            return False
        decision = self.game.decision
        return decision is not None and decision[0] == self.human_seat

    def legal_action_ids(self) -> list[int]:
        low, high = self.game.legal_action_mask
        ids = [i for i in range(64) if (low >> i) & 1]
        ids.extend(64 + i for i in range(51) if (high >> i) & 1)
        return ids

    def is_legal(self, action: int) -> bool:
        return 0 <= action < bm.ACTION_SPACE_SIZE and action in self.legal_action_ids()

    def _relative_seat(self, seat: int) -> int:
        return (seat - self.human_seat) & 3

    def _relative_values(self, values) -> tuple:
        return tuple(values[(self.human_seat + relative) & 3] for relative in range(4))

    def view(self) -> dict:
        """Return UI state with every seat and seat-indexed value viewer-relative."""
        g = self.game
        g.observe_into(
            self.human_seat,
            self._obs_tiles,
            self._obs_melds,
            self._obs_river,
            self._obs_meta,
        )
        meta = self._obs_meta
        decision = g.decision
        own_hand = self._obs_tiles[0].copy()
        locked = [self._obs_tiles[2 + i].copy() for i in range(4)]
        unlocked_hand = own_hand.astype(np.int16) - locked[0].astype(np.int16)
        unlocked_hand = np.maximum(unlocked_hand, 0).astype(np.uint8)
        hand_counts = [int(x) for x in meta[24:28]]
        unlocked_hand_counts = [
            max(0, count - int(locked[i].sum())) for i, count in enumerate(hand_counts)
        ]
        return {
            "phase": g.phase,
            "actor": (
                self._relative_seat(int(decision[0])) if decision is not None else None
            ),
            "scores": self._relative_values(g.scores()),
            "missing": self._relative_values(g.missing_suits()),
            "dealer": self._relative_seat(g.dealer),
            "direction": int(meta[3]),
            "wall_remaining": int(meta[4]),
            "draw_tile": int(meta[5]),
            "response_source": int(meta[7]),
            "response_tile": int(meta[8]),
            "response_flags": int(meta[29]),
            # concealed includes previously won structures. The UI must use
            # unlocked_hand for active play and render locked separately.
            "own_hand": own_hand,
            "unlocked_hand": unlocked_hand,
            "exchange_selection": self._obs_tiles[1].copy(),
            "locked": locked,
            "discards": [self._obs_tiles[6 + i].copy() for i in range(4)],
            "melds": self._obs_melds.copy(),
            "river": self._obs_river.copy(),
            "hand_counts": hand_counts,
            "unlocked_hand_counts": unlocked_hand_counts,
            "has_won": [int(x) for x in meta[20:24]],
            "rankings": (
                tuple(self._relative_seat(seat) for seat in g.rankings())
                if g.phase == bm.PHASE_FINISHED
                else None
            ),
            "termination_reason": g.termination_reason,
        }
