import pytest

import bloodflow_mahjong as bm

from mahjong.model.controller import GameController
from mahjong.model.tiles import SUIT_NAMES, kind_text
from mahjong.ui.main_window import MainWindow


def _window(app):
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    window = MainWindow(controller)
    window.show()
    window.start(seed=17)
    app.processEvents()
    return controller, window


def _click_exchange(window, controller, app):
    kind = min(a for a in controller.legal_action_ids() if a < 27)
    window._on_own_tile_clicked(kind)
    app.processEvents()


def test_exchange_three_clicks_then_choose_missing(app):
    controller, window = _window(app)
    assert controller.game.phase == bm.PHASE_EXCHANGE
    assert controller.human_must_act()
    assert window._bar_hint.text().startswith("换三张")
    assert not any(button.isVisible() for button in window._action_buttons)
    assert not any(button.isVisible() for button in window._btn_missing)
    for _ in range(3):
        _click_exchange(window, controller, app)
    assert controller.game.phase == bm.PHASE_CHOOSE_MISSING
    assert controller.human_must_act()
    window.close()
    app.processEvents()


def test_illegal_exchange_click_rejected(app):
    controller, window = _window(app)
    legal = set(controller.legal_action_ids())
    bad = next(kind for kind in range(27) if kind not in legal)
    before = int(controller.view()["own_hand"].sum())
    window._on_own_tile_clicked(bad)
    app.processEvents()
    assert int(controller.view()["own_hand"].sum()) == before
    assert controller.game.phase == bm.PHASE_EXCHANGE
    window.close()
    app.processEvents()


def test_choose_missing_button_then_turn(app):
    controller, window = _window(app)
    while controller.game.phase == bm.PHASE_EXCHANGE:
        _click_exchange(window, controller, app)
    assert controller.game.phase == bm.PHASE_CHOOSE_MISSING
    assert any(button.isVisible() for button in window._btn_missing)
    assert not any(button.isVisible() for button in window._action_buttons)
    suit = next(a - 27 for a in controller.legal_action_ids() if 27 <= a < 30)
    window._on_missing(suit)
    app.processEvents()
    assert controller.game.phase == bm.PHASE_TURN
    assert controller.human_must_act()
    assert not any(button.isVisible() for button in window._btn_missing)
    window.close()
    app.processEvents()


@pytest.mark.parametrize(
    ("human_seat", "suit"),
    ((1, 0), (2, 1), (3, 2)),
)
def test_missing_buttons_keep_engine_and_relative_ui_in_sync(app, human_seat, suit):
    controller = GameController(
        human_seat=human_seat,
        difficulties={seat: "简单" for seat in range(4) if seat != human_seat},
    )
    window = MainWindow(controller)
    window.show()
    window.start(seed=17)
    app.processEvents()

    while controller.game.phase == bm.PHASE_EXCHANGE:
        _click_exchange(window, controller, app)

    assert controller.game.phase == bm.PHASE_CHOOSE_MISSING
    assert window._btn_missing[suit].isVisible()
    window._btn_missing[suit].click()
    app.processEvents()

    assert controller.game.missing_suits()[human_seat] == suit
    assert controller.view()["missing"][0] == suit
    assert f"缺:{SUIT_NAMES[suit]}" in window._name_labels[0].text()
    assert f"你定缺: {SUIT_NAMES[suit]}" in window.log_text()

    window.close()
    app.processEvents()


def test_turn_discard_submits_and_rotates(app):
    controller, window = _window(app)
    logs = []
    controller.on_log = logs.append
    while controller.game.phase == bm.PHASE_EXCHANGE:
        _click_exchange(window, controller, app)
    if controller.game.phase == bm.PHASE_CHOOSE_MISSING:
        suit = next(a - 27 for a in controller.legal_action_ids() if 27 <= a < 30)
        window._on_missing(suit)
        app.processEvents()
    assert controller.game.phase == bm.PHASE_TURN
    legal = set(controller.legal_action_ids())
    assert window._btn_hu.isVisible() == (bm.ACTION_HU in legal)
    assert window._btn_concealed_kong.isVisible() == any(60 <= a < 87 for a in legal)
    assert window._btn_added_kong.isVisible() == any(87 <= a < 114 for a in legal)
    assert not any(button.isVisible() for button in window._btn_missing)
    discard = min(a - 30 for a in controller.legal_action_ids() if 30 <= a < 57)
    window._on_own_tile_clicked(discard)
    app.processEvents()
    flat = [line for chunk in logs for line in chunk]
    assert any(line.startswith("你打出") for line in flat)
    assert controller.game.phase == bm.PHASE_TURN
    assert controller.human_must_act()
    window.close()
    app.processEvents()


def test_turn_hu_button_distinguishes_self_draw_and_heavenly_hu(app):
    controller, window = _window(app)
    original_legal_action_ids = controller.legal_action_ids
    controller.legal_action_ids = lambda: [bm.ACTION_HU]
    try:
        turn_view = {"phase": bm.PHASE_TURN, "actor": 0, "draw_tile": 5}
        window._update_action_bar(turn_view)
        assert window._btn_hu.text() == "自摸"
        assert "可自摸" in window._bar_hint.text()
        assert not window._btn_pass.isVisible()

        turn_view["draw_tile"] = -1
        window._update_action_bar(turn_view)
        assert window._btn_hu.text() == "天胡"
        assert "可天胡" in window._bar_hint.text()

        controller.legal_action_ids = lambda: [bm.ACTION_HU, bm.ACTION_PASS]
        response_view = {
            "phase": bm.PHASE_HU_RESPONSE,
            "actor": 0,
            "response_tile": 5,
        }
        window._update_action_bar(response_view)
        assert window._btn_hu.text() == "胡"
        assert window._btn_pass.isVisible()
    finally:
        controller.legal_action_ids = original_legal_action_ids
        window.close()
        app.processEvents()


def _find_kong_turn_game(
    base: int = bm.ACTION_CONCEALED_KONG_OFFSET,
    *,
    minimum_candidates: int = 1,
    seed_limit: int = 3000,
):
    for seed in range(seed_limit):
        game = bm.Game(seed=seed)
        while True:
            if game.phase == bm.PHASE_FINISHED:
                break
            decision = game.decision
            if (
                decision is not None
                and decision[0] == 0
                and decision[1] == bm.PHASE_TURN
            ):
                low, high = game.legal_action_mask
                ids = [i for i in range(64) if (low >> i) & 1]
                ids += [64 + i for i in range(51) if (high >> i) & 1]
                kongs = [a for a in ids if base <= a < base + 27]
                if len(kongs) >= minimum_candidates:
                    return game, kongs
            action = game.simple_rule_action()
            if action is None:
                break
            game.step_id(action)
    raise AssertionError("no seed reaches a seat-0 turn with the requested kong action")


def test_concealed_kong_pending_selection(app):
    game, kongs = _find_kong_turn_game()
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    controller.game = game
    window = MainWindow(controller)
    window.show()
    kind = kongs[0] - 60
    window._pending_kong = "暗杠"
    window._refresh()
    app.processEvents()
    assert window._btn_concealed_kong.isVisible()
    assert window._bar_hint.text().startswith("请点击要暗杠的牌")
    clickable = [
        label.kind
        for label in window._own_hand_labels
        if hasattr(label, "kind") and label.kind == kind
    ]
    assert clickable == [kind] * 4
    window._on_own_tile_clicked(kind)
    app.processEvents()
    assert "暗杠" in window.log_text()
    assert window._pending_kong is None
    window.close()
    app.processEvents()


def test_concealed_kong_single_kind_direct_submit(app):
    game, kongs = _find_kong_turn_game()
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    controller.game = game
    window = MainWindow(controller)
    window.show()
    window._refresh()
    app.processEvents()
    assert window._btn_concealed_kong.isVisible()
    window._on_concealed_kong()
    app.processEvents()
    if len(kongs) == 1:
        assert "暗杠" in window.log_text()
        assert window._pending_kong is None
    window.close()
    app.processEvents()


@pytest.mark.parametrize(
    ("seed_limit", "base", "hint_name"),
    [
        (3000, bm.ACTION_CONCEALED_KONG_OFFSET, "暗杠"),
        (20000, bm.ACTION_ADDED_KONG_OFFSET, "碰杠"),
    ],
)
def test_kong_hint_lists_every_available_tile(app, seed_limit, base, hint_name):
    found, actions = _find_kong_turn_game(base, seed_limit=seed_limit)
    kinds = [action - base for action in actions]
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    controller.game = found
    window = MainWindow(controller)
    window.show()
    window._refresh()
    app.processEvents()

    hint = window._bar_hint.text()
    assert hint_name in hint
    for kind in kinds:
        assert kind_text(kind) in hint
        candidate_labels = window._own_hand_labels + window._locked_labels[0]
        labels = [label for label in candidate_labels if label.kind == kind]
        assert labels
        hinted = [label for label in labels if hint_name in label.toolTip()]
        assert hinted
        assert all("#f9a825" in label.styleSheet() for label in hinted)

    window.close()
    app.processEvents()


def test_multiple_added_kong_candidates_can_be_selected_from_visible_tiles(app):
    game, actions = _find_kong_turn_game(
        bm.ACTION_ADDED_KONG_OFFSET,
        minimum_candidates=2,
        seed_limit=3000,
    )
    kinds = [action - bm.ACTION_ADDED_KONG_OFFSET for action in actions]
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    controller.game = game
    window = MainWindow(controller)
    window.show()
    window._refresh()
    app.processEvents()

    window._on_added_kong()
    app.processEvents()
    assert window._pending_kong == "碰杠"
    assert all(kind_text(kind) in window._bar_hint.text() for kind in kinds)

    candidates = window._own_hand_labels + window._locked_labels[0]
    clickable_kinds = {
        label.kind for label in candidates if "选择碰杠" in label.toolTip()
    }
    assert clickable_kinds == set(kinds)

    window._on_own_tile_clicked(kinds[0])
    app.processEvents()
    assert window._pending_kong is None
    # A legal added-kong action can be robbed by another player before the
    # meld event is committed.
    assert "碰杠" in window.log_text() or "抢杠" in window.log_text()

    window.close()
    app.processEvents()


def _find_response_game(target: int, limit: int = 4000):
    for seed in range(limit):
        game = bm.Game(seed=seed)
        while True:
            if game.phase == bm.PHASE_FINISHED:
                break
            decision = game.decision
            if decision is not None and decision[0] == 0 and decision[1] == target:
                return game
            action = game.simple_rule_action()
            if action is None:
                break
            game.step_id(action)
    raise AssertionError(f"no seed reaches seat-0 phase {target}")


@pytest.mark.parametrize("target", [bm.PHASE_HU_RESPONSE, bm.PHASE_MELD_RESPONSE])
def test_response_buttons_match_legal_mask(app, target):
    game = _find_response_game(target)
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    controller.game = game
    window = MainWindow(controller)
    window.show()
    window._refresh()
    app.processEvents()
    legal = set(controller.legal_action_ids())
    assert window._btn_pass.isVisible() == (bm.ACTION_PASS in legal)
    assert window._btn_hu.isVisible() == (bm.ACTION_HU in legal)
    assert window._btn_pong.isVisible() == (bm.ACTION_PONG in legal)
    assert window._btn_exposed_kong.isVisible() == (bm.ACTION_EXPOSED_KONG in legal)
    assert not window._btn_concealed_kong.isVisible()
    assert not window._btn_added_kong.isVisible()
    assert not any(button.isVisible() for button in window._btn_missing)
    window.close()
    app.processEvents()


def test_action_buttons_hidden_outside_own_turn(app):
    controller, window = _window(app)
    assert not any(button.isVisible() for button in window._action_buttons)
    assert not any(button.isVisible() for button in window._btn_missing)
    assert window._bar_hint.text().startswith("换三张")
    window.close()
    app.processEvents()


def test_full_game_played_through_ui_handlers(app):
    controller, window = _window(app)
    steps = 0
    while controller.game.phase != bm.PHASE_FINISHED and steps < 2000:
        if controller.human_must_act():
            phase = controller.game.phase
            legal = controller.legal_action_ids()
            if phase == bm.PHASE_EXCHANGE:
                window._on_own_tile_clicked(min(a for a in legal if a < 27))
            elif phase == bm.PHASE_CHOOSE_MISSING:
                window._on_missing(next(a - 27 for a in legal if 27 <= a < 30))
            elif phase == bm.PHASE_TURN:
                discards = [a - 30 for a in legal if 30 <= a < 57]
                assert discards, f"turn with no discardable tile: {legal}"
                window._on_own_tile_clicked(min(discards))
            elif phase in (bm.PHASE_HU_RESPONSE, bm.PHASE_MELD_RESPONSE):
                window._on_pass()
            steps += 1
        app.processEvents()
    assert controller.game.phase == bm.PHASE_FINISHED, controller.game.phase
    assert controller.game.termination_reason is not None
    assert sorted(controller.game.rankings()) == [0, 1, 2, 3]
    window.close()
    app.processEvents()
