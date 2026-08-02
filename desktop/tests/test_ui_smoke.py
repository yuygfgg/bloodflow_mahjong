import os
import sys
import pathlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from PySide6.QtWidgets import QApplication

from mahjong.model.controller import GameController
from mahjong.ui.main_window import MainWindow
from mahjong.ui.tile_label import resolve_family, tile_label


def test_tile_label_factory(app):
    label = tile_label(0)
    assert label.text()
    back = tile_label(0, back=True)
    assert back.text()
    assert resolve_family() is None or isinstance(resolve_family(), str)


def test_opening_state_renders(app):
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    window = MainWindow(controller)
    window.start(seed=7)
    app.processEvents()
    assert window.own_hand_label_count() == 14
    assert window.ai_hand_label_counts() == {1: 13, 2: 13, 3: 13}
    assert window.own_hand_label_count() == 14
    assert window.log_text().startswith("牌局开始")
    assert "张" in window.info_text()
    window.close()
    app.processEvents()


def test_window_plays_full_game(app):
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    controller.human_proxy = lambda game: game.simple_rule_action()
    window = MainWindow(controller)
    window.start(seed=9)
    app.processEvents()
    assert controller.game.phase == 5
    assert "对局结束" in window.log_text()
    assert "领先" in "".join(l.text() for l in window._score_labels)
    window.close()
    app.processEvents()


def test_refresh_after_steps_updates_seats(app):
    controller = GameController(difficulties={1: "简单", 2: "简单", 3: "简单"})
    window = MainWindow(controller)
    window.start(seed=11)
    app.processEvents()
    controller.human_proxy = lambda game: game.simple_rule_action()
    for _ in range(40):
        if controller.human_must_act():
            controller.submit(controller.legal_action_ids()[0])
        app.processEvents()
    assert controller.game.phase >= 2 or controller.game.phase == 5
    window.close()
    app.processEvents()
