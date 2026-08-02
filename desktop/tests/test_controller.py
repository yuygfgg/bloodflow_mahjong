import bloodflow_mahjong as bm
import numpy as np

from mahjong.model.controller import GameController, format_event
from mahjong.model.policy import make_policy
from mahjong.model.tiles import (
    BACK_GLYPH,
    SUIT_NAMES,
    kind_suit,
    kind_text,
    kind_to_glyph,
)


def test_all_kinds_glyph_mapping():
    for kind in range(27):
        suit, rank = kind_suit(kind), kind % 9
        assert kind_to_glyph(kind) == chr(0x1F007 + suit * 9 + rank)
        assert kind_text(kind) == f"{rank + 1}{SUIT_NAMES[suit]}"
    assert kind_to_glyph(0) == "\U0001f007"
    assert kind_to_glyph(8) == "\U0001f00f"
    assert kind_to_glyph(9) == "\U0001f010"
    assert kind_to_glyph(17) == "\U0001f018"
    assert kind_to_glyph(18) == "\U0001f019"
    assert kind_to_glyph(26) == "\U0001f021"
    assert BACK_GLYPH == "\U0001f02b"


def test_policy_construction():
    assert make_policy("简单") is not None
    assert make_policy("标准") is not None
    assert make_policy("困难") is not None
    try:
        make_policy("地狱")
        assert False
    except ValueError:
        pass


def test_full_game_runs_headless():
    logs = []
    controller = GameController(
        difficulties={1: "简单", 2: "简单", 3: "简单"}, on_log=logs.append
    )
    controller.human_proxy = lambda game: game.simple_rule_action()
    controller.start(seed=123)
    assert controller.game.phase == bm.PHASE_FINISHED
    assert controller.game.termination_reason is not None
    assert sum(controller.game.scores()) == 40000
    assert sorted(controller.game.rankings()) == [0, 1, 2, 3]
    assert controller.game.event_dropped == 0
    assert any("牌局" in line for chunk in logs for line in chunk)


def test_illegal_action_rejected_without_state_change():
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    controller.start(seed=1)
    assert controller.human_must_act()
    phase_before = controller.game.phase
    legal = set(controller.legal_action_ids())
    bad = next(i for i in range(bm.ACTION_SPACE_SIZE) if i not in legal)
    assert controller.submit(bad) is False
    assert controller.game.phase == phase_before
    assert controller.human_must_act()
    good = next(iter(legal))
    assert controller.submit(good) is True


def test_view_reflects_opening_state():
    controller = GameController(difficulties={1: "标准", 2: "标准", 3: "标准"})
    controller.start(seed=42)
    view = controller.view()
    assert view["phase"] == bm.PHASE_EXCHANGE
    assert view["actor"] == 0
    assert view["scores"] == (10000, 10000, 10000, 10000)
    assert view["hand_counts"] == [14, 13, 13, 13]
    assert int(view["own_hand"].sum()) == 14
    assert int(view["unlocked_hand"].sum()) == 14
    assert view["unlocked_hand_counts"] == [14, 13, 13, 13]
    assert view["direction"] in (1, 2, 3)
    assert view["wall_remaining"] == 108 - 14 - 13 * 3
    assert view["melds"].shape == (4, 4, 3)
    assert view["river"].shape == (108, 2)


def test_view_uses_relative_seats_for_all_player_fields():
    human_seat = 2
    controller = GameController(
        human_seat=human_seat,
        difficulties={seat: "简单" for seat in range(4) if seat != human_seat},
    )
    controller.start(seed=42)

    view = controller.view()

    assert view["actor"] == 0
    assert view["dealer"] == 2
    assert view["scores"] == tuple(
        controller.game.scores()[(human_seat + relative_seat) & 3]
        for relative_seat in range(4)
    )
    assert view["missing"] == (-1, -1, -1, -1)


def test_view_separates_locked_wins_from_active_hands():
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    controller.game = bm.Game(seed=0)

    while controller.game.phase != bm.PHASE_FINISHED:
        view = controller.view()
        if int(view["locked"][0].sum()) > 0:
            break
        controller.game.step_id(controller.game.simple_rule_action())
    else:
        raise AssertionError("seed did not produce a bloodflow win")

    concealed = int(view["own_hand"].sum())
    locked = int(view["locked"][0].sum())
    active = int(view["unlocked_hand"].sum())
    assert locked > 0
    assert active == concealed - locked
    assert view["unlocked_hand_counts"][0] == active
    for relative_seat in range(4):
        assert view["unlocked_hand_counts"][relative_seat] == (
            view["hand_counts"][relative_seat]
            - int(view["locked"][relative_seat].sum())
        )


def test_view_reveals_only_an_opponents_winning_tiles():
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    controller.game = bm.Game(seed=0)

    while controller.game.phase != bm.PHASE_FINISHED:
        controller.game.step_id(controller.game.simple_rule_action())
        view = controller.view()
        winner = next(
            (
                relative_seat
                for relative_seat in range(1, 4)
                if view["has_won"][relative_seat]
            ),
            None,
        )
        if winner is not None:
            break
    else:
        raise AssertionError("seed did not produce an opponent win")

    assert int(view["locked"][winner].sum()) == 1
    assert view["unlocked_hand_counts"][winner] == view["hand_counts"][winner] - 1


def test_format_event_lines():
    e = np.array([bm.EventKind.DISCARD, 1, -1, 9, 0, 0, 0, -1], dtype=np.int32)
    assert format_event(e) == "下家打出 1条"
    e = np.array(
        [bm.EventKind.HU, 0, 2, 14, bm.EventFlag.SELF_DRAW, 4, 3, -1], dtype=np.int32
    )
    assert format_event(e) == "你和牌 6条 ×4 (自摸)"
    e = np.array([bm.EventKind.DRAW, 0, -1, 18, 0, 0, 0, -1], dtype=np.int32)
    assert format_event(e) == "你摸到 1筒"
    e = np.array([bm.EventKind.DRAW, 2, -1, -1, 0, 0, 0, -1], dtype=np.int32)
    assert format_event(e) == "对家摸牌"
    e = np.array([bm.EventKind.MELD, 3, 0, 5, 0, 0, 0, -1], dtype=np.int32)
    assert format_event(e) == "上家碰 6万"
    e = np.array(
        [
            bm.EventKind.ACTION,
            2,
            -1,
            14,
            0,
            bm.ACTION_ADDED_KONG_OFFSET + 14,
            bm.PHASE_TURN,
            -1,
        ],
        dtype=np.int32,
    )
    assert format_event(e) == "对家声明碰杠 6条"
    assert format_event(np.zeros(8, dtype=np.int32)) is None


def test_viewer_does_not_see_opponent_draws():
    logs = []
    controller = GameController(
        difficulties={1: "简单", 2: "简单", 3: "简单"}, on_log=logs.append
    )
    controller.human_proxy = lambda game: game.simple_rule_action()
    controller.start(seed=7)
    for line in (line for chunk in logs for line in chunk):
        assert "下家摸牌" in line or "摸到" not in line or line.startswith("你摸到")
