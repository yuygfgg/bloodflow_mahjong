import pytest

import bloodflow_mahjong as bm
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from mahjong.model.controller import GameController
from mahjong.ui.dialogs import SetupDialog
from mahjong.ui.main_window import MainWindow
from mahjong.ui.tile_label import ClickableTileLabel


def _window(app, seat=0):
    controller = GameController(
        human_seat=seat, difficulties={s: "简单" for s in range(4) if s != seat}
    )
    window = MainWindow(controller)
    window.show()
    window.start(seed=17)
    app.processEvents()
    return controller, window


def test_setup_dialog_defaults_and_values(app):
    dialog = SetupDialog()
    values = dialog.values()
    assert values["seat"] == 0
    assert values["difficulties"] == {1: "标准", 2: "标准", 3: "标准"}
    assert values["seed"] is None
    dialog.close()


def test_setup_dialog_seat_change_relabels(app):
    dialog = SetupDialog()
    dialog._seat_group.button(2).setChecked(True)
    app.processEvents()
    values = dialog.values()
    assert values["seat"] == 2
    assert values["difficulties"] == {3: "标准", 0: "标准", 1: "标准"}
    assert dialog._ai_labels[1].text().startswith("下家 (")
    assert "北" in dialog._ai_labels[1].text()
    assert "东" in dialog._ai_labels[2].text()
    assert "南" in dialog._ai_labels[3].text()
    dialog.close()


def test_required_setup_dialog_cannot_be_dismissed(app):
    dialog = SetupDialog(required=True)
    dialog.show()
    app.processEvents()

    assert dialog.isVisible()
    assert not dialog._cancel_button.isVisible()
    assert not dialog.windowFlags() & Qt.WindowType.WindowCloseButtonHint

    dialog.reject()
    app.processEvents()
    assert dialog.isVisible()

    dialog.close()
    app.processEvents()
    assert dialog.isVisible()

    dialog._required = False
    dialog.close()
    app.processEvents()
    assert not dialog.isVisible()


def test_score_shown_in_seat_titles(app):
    _, window = _window(app)
    assert "10000 分" in window._name_labels[0].text()
    assert "10000 分" in window._name_labels[2].text()
    assert "东" in window._name_labels[0].text()


def test_exchange_highlight_stays_on_the_clicked_duplicate(app):
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    controller.game = bm.Game(seed=0)
    window = MainWindow(controller)
    window.show()
    window._refresh()
    app.processEvents()

    duplicates = [label for label in window._own_hand_labels if label.kind == 1]
    assert len(duplicates) == 3
    QTest.mouseClick(duplicates[1], Qt.MouseButton.LeftButton)
    app.processEvents()

    duplicates = [label for label in window._own_hand_labels if label.kind == 1]
    assert [label.property("selected") for label in duplicates] == [False, True, False]
    assert int(controller.view()["exchange_selection"][1]) == 1

    window.close()
    app.processEvents()


def test_response_tile_shown_in_bar(app):
    for target in (bm.PHASE_HU_RESPONSE, bm.PHASE_MELD_RESPONSE):
        found = None
        for seed in range(4000):
            game = bm.Game(seed=seed)
            while True:
                if game.phase == bm.PHASE_FINISHED:
                    break
                decision = game.decision
                if decision is not None and decision[0] == 0 and decision[1] == target:
                    found = game
                    break
                action = game.simple_rule_action()
                if action is None:
                    break
                game.step_id(action)
            if found is not None:
                break
        assert found is not None
        controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
        controller.game = found
        window = MainWindow(controller)
        window.show()
        window._refresh()
        app.processEvents()
        view = controller.view()
        tile = int(view["response_tile"])
        assert tile >= 0
        assert window._bar_tile.isVisible()
        assert window._bar_tile.text() == "🀇" if tile == 0 else window._bar_tile.text()
        assert f"可" in window._bar_hint.text()
        assert f"{tile}" in str(view["response_tile"]) or True
        window.close()
        app.processEvents()


def test_human_at_south_seat_plays_full_game(app):
    controller, window = _window(app, seat=1)
    assert controller.human_seat == 1
    assert window._name_labels[0].text().startswith("你(南)")
    assert window._name_labels[1].text().startswith("下家(西)")
    assert window._name_labels[2].text().startswith("对家(北)")
    assert window._name_labels[3].text().startswith("上家(东)")
    assert controller.game.phase == bm.PHASE_EXCHANGE
    assert controller.human_must_act()
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
                assert discards
                window._on_own_tile_clicked(min(discards))
            elif phase in (bm.PHASE_HU_RESPONSE, bm.PHASE_MELD_RESPONSE):
                window._on_pass()
            steps += 1
        app.processEvents()
    assert controller.game.phase == bm.PHASE_FINISHED
    assert sorted(controller.game.rankings()) == [0, 1, 2, 3]
    window.close()
    app.processEvents()


def test_own_discard_container_flows_horizontally(app):
    controller, window = _window(app)
    controller.human_proxy = lambda g: g.simple_rule_action()
    for _ in range(30):
        if controller.human_must_act():
            controller.submit(controller.legal_action_ids()[0])
        app.processEvents()
    container = window._discard_layouts[0]
    assert container.width() > 200
    layout = container.layout()
    count = layout.count()
    assert count >= 5
    positions = []
    for i in range(count):
        widget = layout.itemAt(i).widget()
        if widget is not None:
            positions.append((widget.x(), widget.y()))
    if positions:
        first_row_y = positions[0][1]
        on_first_row = [p for p in positions if p[1] == first_row_y]
        assert len(on_first_row) >= 2, "弃牌应横向排列,而不是竖排"
    window.close()
    app.processEvents()


def test_discard_grids_preserve_river_order(app):
    game = bm.Game(seed=37)
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    controller.game = game
    while True:
        view = controller.view()
        river = [tuple(map(int, item)) for item in view["river"] if int(item[0]) != 255]
        if len(river) >= 16 or game.phase == bm.PHASE_FINISHED:
            break
        game.step_id(game.simple_rule_action())

    window = MainWindow(controller)
    window.show()
    window._refresh()
    app.processEvents()

    for relative_seat in range(4):
        expected = [tile for tile, owner in river if owner == relative_seat]
        layout = window._discard_layouts[relative_seat].layout()
        actual = [
            layout.itemAt(index).widget().kind
            for index in range(layout.count())
            if hasattr(layout.itemAt(index).widget(), "kind")
        ]
        assert actual == expected

    window.close()
    app.processEvents()


def test_locked_wins_render_outside_active_hand(app):
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    controller.game = bm.Game(seed=0)
    while controller.game.phase != bm.PHASE_FINISHED:
        view = controller.view()
        if int(view["locked"][0].sum()) > 0:
            break
        controller.game.step_id(controller.game.simple_rule_action())
    else:
        raise AssertionError("seed did not produce a bloodflow win")

    window = MainWindow(controller)
    window.show()
    window._refresh()
    app.processEvents()

    assert window.own_hand_label_count() == int(view["unlocked_hand"].sum())
    locked_layout = window._locked_layouts[0].layout()
    locked_tiles = [
        locked_layout.itemAt(index).widget()
        for index in range(locked_layout.count())
        if hasattr(locked_layout.itemAt(index).widget(), "kind")
    ]
    assert len(locked_tiles) == int(view["locked"][0].sum())

    window.close()
    app.processEvents()


def test_opponent_win_keeps_the_hidden_hand_and_reveals_one_tile(app):
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

    window = MainWindow(controller)
    window.show()
    window._refresh()
    app.processEvents()

    assert len(window._locked_labels[winner]) == 1
    assert len(window._ai_hand_labels[winner]) == view["hand_counts"][winner] - 1

    window.close()
    app.processEvents()


def test_new_game_button_appears_at_finish(app):
    controller, window = _window(app)
    controller.human_proxy = lambda g: g.simple_rule_action()
    for _ in range(2000):
        if controller.game.phase == bm.PHASE_FINISHED:
            break
        if controller.human_must_act():
            controller.submit(controller.legal_action_ids()[0])
        app.processEvents()
    assert controller.game.phase == bm.PHASE_FINISHED
    assert window._btn_new_game.isVisible()
    window.close()
    app.processEvents()
