from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import bloodflow_mahjong as bm

from ..model.controller import GameController
from ..model.tiles import (
    DIRECTION_NAMES,
    MELD_NAMES,
    PHASE_NAMES,
    SEAT_NAMES,
    SUIT_NAMES,
    count_to_hand,
    kind_text,
    kind_to_glyph,
)
from .dialogs import SetupDialog, WIND_NAMES
from .tile_label import tile_label

_OWN_SIZE = (50, 70)
_OWN_FONT = 24
_AI_SIZE = (30, 42)
_AI_FONT = 15
_PUBLIC_SIZE = (30, 42)
_PUBLIC_FONT = 15
_LOCKED_SIZE = (26, 36)
_LOCKED_FONT = 13

_QSS = """
QMainWindow {
    background: #071d17;
}
QWidget#centralRoot {
    background: #0b2b20;
}
QWidget#tableFelt {
    background: #176149;
    border: 10px solid #3a2b1d;
    border-radius: 16px;
}
QFrame#seatFrame {
    background: transparent;
    border: none;
}
QFrame#seatBadge {
    background: rgba(7, 37, 28, 0.78);
    border: 1px solid rgba(238, 201, 123, 0.48);
    border-radius: 6px;
    min-height: 26px;
}
QFrame#seatBadgeActive {
    background: rgba(89, 68, 23, 0.92);
    border: 2px solid #efc76b;
    border-radius: 6px;
    min-height: 26px;
}
QFrame#publicZone {
    background: rgba(5, 43, 32, 0.28);
    border: 1px solid rgba(201, 231, 199, 0.16);
    border-radius: 6px;
}
QLabel#zoneCaption {
    color: rgba(221, 235, 222, 0.72);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    padding-left: 2px;
}
QFrame#infoFrame {
    background: #132a26;
    border: 2px solid #d4a95e;
    border-radius: 10px;
    min-width: 188px;
    min-height: 188px;
}
QFrame#sideFrame {
    background: #0c211b;
    border: 1px solid rgba(196, 163, 90, 0.38);
    border-radius: 8px;
}
QFrame#actionBar {
    background: rgba(7, 32, 24, 0.9);
    border: 1px solid rgba(229, 190, 105, 0.48);
    border-radius: 8px;
}
QLabel#seatName {
    color: #f8e6bd;
    font-weight: 700;
    font-size: 12px;
}
QLabel#seatInfo {
    color: #bbd6c8;
    font-size: 12px;
}
QLabel#phaseTitle {
    color: #ffe4a4;
    font-size: 14px;
    font-weight: 700;
}
QLabel#remainCaption {
    color: #c5b17d;
    font-size: 12px;
    font-weight: 600;
}
QLabel#remainValue {
    color: #ffbd54;
    font-size: 34px;
    font-weight: 800;
}
QLabel#windLabel {
    color: #a4b5ad;
    font-size: 13px;
    font-weight: 700;
}
QLabel#windActive {
    color: #ffbf61;
    font-size: 14px;
    font-weight: 800;
}
QLabel#barHint {
    color: #f7dfac;
    font-size: 14px;
    font-weight: 700;
}
QLabel#meldTag {
    color: #f4cd78;
    font-size: 10px;
    font-weight: 700;
    padding: 1px 3px;
    background: rgba(0, 0, 0, 0.26);
    border-radius: 3px;
}
QLabel#sideTitle {
    color: #e9c97d;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
QLabel#sideText {
    color: #d5e4da;
    font-size: 12px;
    padding: 5px 6px;
    background: rgba(255, 255, 255, 0.045);
    border-radius: 4px;
}
QLabel#sideTextLead {
    color: #ffe082;
    font-size: 12px;
    font-weight: 700;
    padding: 5px 6px;
    background: rgba(255, 193, 7, 0.13);
    border: 1px solid rgba(255, 193, 7, 0.38);
    border-radius: 4px;
}
QPlainTextEdit#logView {
    background: #081812;
    color: #c9d8cf;
    border: 1px solid rgba(196, 163, 90, 0.25);
    border-radius: 5px;
    font-size: 12px;
    padding: 4px;
}
QPushButton {
    background: #30463d;
    color: #f2efe6;
    border: 1px solid #6d7d74;
    border-radius: 7px;
    padding: 7px 14px;
    font-size: 14px;
    font-weight: 700;
    min-height: 32px;
}
QPushButton:hover:enabled {
    background: #416056;
    border-color: #9fb0a7;
}
QPushButton:pressed:enabled {
    background: #1c2924;
}
QPushButton:disabled {
    color: #7d8882;
    background: #24302c;
    border-color: #3a4641;
}
QPushButton#dangerBtn {
    background: #b84235;
    color: white;
    border: 1px solid #ff8a80;
    border-radius: 7px;
    padding: 7px 16px;
    font-size: 16px;
    min-width: 70px;
}
QPushButton#dangerBtn:hover:enabled {
    background: #d55446;
}
QPushButton#okBtn {
    background: #2f7651;
    color: white;
    border: 1px solid #81c784;
    border-radius: 7px;
    min-width: 64px;
}
QPushButton#okBtn:hover:enabled {
    background: #3b9063;
}
QPushButton#passBtn {
    background: #4d5b5e;
    color: #f3f4f6;
    border: 1px solid #9ca3af;
    border-radius: 7px;
    min-width: 64px;
}
QPushButton#passBtn:hover:enabled {
    background: #637277;
}
QPushButton#missingBtn {
    background: #345f91;
    color: white;
    border: 1px solid #90caf9;
    border-radius: 7px;
    min-width: 64px;
}
QPushButton#missingBtn:hover:enabled {
    background: #477ab1;
}
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: rgba(196, 163, 90, 0.35);
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""


class MainWindow(QMainWindow):
    def __init__(self, controller: GameController | None = None) -> None:
        super().__init__()
        self.controller: GameController | None = controller
        self._human_seat = controller.human_seat if controller is not None else 0
        self._last_settings: dict = {}
        self.setWindowTitle("血流麻将 · 本地对局")
        self.resize(1280, 820)
        self.setStyleSheet(_QSS)
        self._own_hand_labels: list[QLabel] = []
        self._ai_hand_labels: dict[int, list[QLabel]] = {}
        self._seat_frames = {}
        self._badge_frames = {}
        self._hand_rows = {}
        self._name_labels = {}
        self._discard_layouts: dict[int, QWidget] = {}
        self._meld_layouts: dict[int, QWidget] = {}
        self._locked_layouts: dict[int, QWidget] = {}
        self._locked_labels: dict[int, list[QLabel]] = {}
        self._wind_labels: dict[str, QLabel] = {}
        self._pending_kong: str | None = None
        self._exchange_selected_slots: set[tuple[int, int]] = set()
        self._build_ui()
        if controller is not None:
            self.set_controller(controller)

    def set_controller(self, controller: GameController) -> None:
        """Replace the controller and rebind callbacks for a new match."""
        if self.controller is not None:
            self.controller.on_state_changed = None
            self.controller.on_log = None
            self.controller.on_finished = None
        self.controller = controller
        self._human_seat = controller.human_seat
        self._pending_kong = None
        self._exchange_selected_slots.clear()
        controller.on_state_changed = self._refresh
        controller.on_log = self._append_log
        controller.on_finished = self._on_finished
        self._log.clear()

    def start(self, seed: int | None = None) -> None:
        self._exchange_selected_slots.clear()
        if self.controller is None:
            self._open_setup(seed)
        else:
            self.controller.start(seed)

    def _open_setup(self, seed: int | None) -> None:
        dialog = SetupDialog(
            self,
            defaults=self._last_settings,
            required=self.controller is None,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        settings = dialog.values()
        self._last_settings = settings
        controller = GameController(
            human_seat=settings["seat"], difficulties=settings["difficulties"]
        )
        self.set_controller(controller)
        controller.start(
            seed=settings["seed"] if settings["seed"] is not None else seed
        )

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralRoot")
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        table_shell = QWidget()
        table_shell.setObjectName("tableFelt")
        table_layout = QVBoxLayout(table_shell)
        table_layout.setSpacing(7)
        table_layout.setContentsMargins(14, 12, 14, 12)

        board = QWidget()
        board_layout = QGridLayout(board)
        board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.setHorizontalSpacing(8)
        board_layout.setVerticalSpacing(5)

        # Widget keys are relative seats: self, right, opposite, left.
        # This keeps the human player at the bottom for every absolute seat.
        self._top_frame = self._make_seat_frame(2)
        self._left_frame = self._make_seat_frame(3)
        self._right_frame = self._make_seat_frame(1)
        self._bottom_frame = self._make_seat_frame(0)

        board_layout.addWidget(self._top_frame, 0, 1)
        board_layout.addWidget(self._left_frame, 1, 0)
        board_layout.addWidget(self._build_center_stack(), 1, 1)
        board_layout.addWidget(self._right_frame, 1, 2)
        board_layout.addWidget(self._bottom_frame, 2, 0, 1, 3)
        board_layout.setColumnStretch(0, 2)
        board_layout.setColumnStretch(1, 5)
        board_layout.setColumnStretch(2, 2)
        board_layout.setRowStretch(0, 1)
        board_layout.setRowStretch(1, 4)
        board_layout.setRowStretch(2, 2)

        table_layout.addWidget(board, 1)
        table_layout.addWidget(self._build_action_bar())

        root.addWidget(table_shell, 1)
        root.addWidget(self._build_side_frame(), 0)

        self.setCentralWidget(central)

    def _make_seat_frame(self, seat: int) -> QFrame:
        """Build one physical seat area, keyed by relative seat."""
        frame = QFrame()
        frame.setObjectName("seatFrame")
        root = QHBoxLayout(frame) if seat in (1, 3) else QVBoxLayout(frame)
        root.setSpacing(5)
        root.setContentsMargins(3, 2, 3, 2)

        hand = QGridLayout() if seat in (1, 3) else QHBoxLayout()
        hand.setSpacing(2)
        hand.setContentsMargins(0, 0, 0, 0)

        discard_block, discards = self._zone_block("牌河")
        locked_block, locked = self._zone_block("和牌")
        meld_block, melds = self._zone_block("副露")
        name = self._make_name_badge(seat, Qt.AlignmentFlag.AlignCenter)
        name_label = name.findChild(QLabel, "seatName")

        if seat == 2:
            root.addLayout(hand)
            root.addWidget(name)
            public_row = QHBoxLayout()
            public_row.setSpacing(6)
            public_row.addStretch(1)
            public_row.addWidget(discard_block, 3)
            public_row.addWidget(locked_block, 3)
            public_row.addWidget(meld_block, 2)
            public_row.addStretch(1)
            root.addLayout(public_row)
        elif seat == 3:
            root.addLayout(hand)
            info = QVBoxLayout()
            info.setSpacing(5)
            info.addWidget(name)
            info.addWidget(discard_block, 3)
            exposed_row = QHBoxLayout()
            exposed_row.setSpacing(5)
            exposed_row.addWidget(locked_block, 3)
            exposed_row.addWidget(meld_block, 2)
            info.addLayout(exposed_row, 3)
            root.addLayout(info)
        elif seat == 1:
            info = QVBoxLayout()
            info.setSpacing(5)
            info.addWidget(name)
            info.addWidget(discard_block, 3)
            exposed_row = QHBoxLayout()
            exposed_row.setSpacing(5)
            exposed_row.addWidget(locked_block, 3)
            exposed_row.addWidget(meld_block, 2)
            info.addLayout(exposed_row, 3)
            root.addLayout(info)
            root.addLayout(hand)
        else:
            info_row = QHBoxLayout()
            info_row.setSpacing(6)
            info_row.addStretch(1)
            info_row.addWidget(meld_block, 2)
            info_row.addWidget(locked_block, 3)
            info_row.addWidget(discard_block, 3)
            info_row.addStretch(1)
            root.addLayout(info_row)
            root.addWidget(name)
            root.addLayout(hand)

        self._seat_frames[seat] = frame
        self._badge_frames[seat] = name
        self._hand_rows[seat] = hand
        self._name_labels[seat] = name_label
        self._discard_layouts[seat] = discards
        self._meld_layouts[seat] = melds
        self._locked_layouts[seat] = locked
        self._ai_hand_labels[seat] = []
        return frame

    def _make_name_badge(self, seat: int, align) -> QFrame:
        badge = QFrame()
        badge.setObjectName("seatBadge")
        badge.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(badge)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(0)
        name = QLabel(self._seat_title(seat))
        name.setObjectName("seatName")
        name.setAlignment(align)
        layout.addWidget(name)
        return badge

    def _zone_block(self, title: str) -> tuple[QWidget, QWidget]:
        block = QWidget()
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        caption = QLabel(title)
        caption.setObjectName("zoneCaption")
        layout.addWidget(caption)
        container = self._public_container()
        container.setToolTip(f"{title}区")
        layout.addWidget(container, 1)
        return block, container

    def _public_container(self) -> QFrame:
        """Return a public-tile surface with an explicit grid layout."""
        container = QFrame()
        container.setObjectName("publicZone")
        container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        layout = QGridLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setHorizontalSpacing(2)
        layout.setVerticalSpacing(2)
        return container

    def _build_center_stack(self) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._build_info_frame(), 0, Qt.AlignmentFlag.AlignCenter)
        return wrap

    def _build_info_frame(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("infoFrame")
        frame.setFixedSize(214, 214)
        layout = QGridLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)

        north = QLabel("北")
        west = QLabel("西")
        east = QLabel("东")
        south = QLabel("南")
        for label in (north, west, east, south):
            label.setObjectName("windLabel")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        center = QVBoxLayout()
        center.setSpacing(0)
        center.setAlignment(Qt.AlignmentFlag.AlignCenter)
        remain_caption = QLabel("余")
        remain_caption.setObjectName("remainCaption")
        remain_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        remain_value = QLabel("—")
        remain_value.setObjectName("remainValue")
        remain_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        phase = QLabel("阶段")
        phase.setObjectName("phaseTitle")
        phase.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(remain_caption)
        center.addWidget(remain_value)
        center.addSpacing(2)
        center.addWidget(phase)

        meta = QLabel("—")
        meta.setObjectName("seatInfo")
        meta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meta.setWordWrap(True)

        layout.addWidget(north, 0, 1)
        layout.addWidget(west, 1, 0)
        layout.addLayout(center, 1, 1)
        layout.addWidget(east, 1, 2)
        layout.addWidget(south, 2, 1)
        layout.addWidget(meta, 3, 0, 1, 3)

        self._phase_title = phase
        self._remain_value = remain_value
        self._center_meta = meta
        self._wind_labels = {"north": north, "west": west, "east": east, "south": south}
        # Keep legacy info_labels keys used by info_text().
        self._info_labels = {
            "wall": remain_value,
            "direction": meta,
            "dealer": meta,
        }
        return frame

    def _build_side_frame(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("sideFrame")
        layout = QVBoxLayout(frame)
        layout.setSpacing(7)
        layout.setContentsMargins(10, 10, 10, 10)
        frame.setMinimumWidth(214)
        frame.setMaximumWidth(236)

        title = QLabel("积分")
        title.setObjectName("sideTitle")
        layout.addWidget(title)

        self._score_labels = []
        for _seat in range(4):
            label = QLabel()
            label.setObjectName("sideText")
            layout.addWidget(label)
            self._score_labels.append(label)

        log_title = QLabel("事件")
        log_title.setObjectName("sideTitle")
        layout.addWidget(log_title)

        self._log = QPlainTextEdit()
        self._log.setObjectName("logView")
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(300)
        layout.addWidget(self._log, 1)
        return frame

    def _build_action_bar(self) -> QFrame:
        """Action strip: hint + contextual buttons for the current decision."""
        frame = QFrame()
        frame.setObjectName("actionBar")
        layout = QHBoxLayout(frame)
        layout.setSpacing(7)
        layout.setContentsMargins(10, 7, 10, 7)
        frame.setMinimumHeight(50)

        self._bar_hint = QLabel("")
        self._bar_hint.setObjectName("barHint")
        layout.addWidget(self._bar_hint)
        self._bar_tile = tile_label(0, size=_OWN_SIZE, font_pt=_OWN_FONT)
        self._bar_tile.hide()
        layout.addWidget(self._bar_tile)
        layout.addStretch(1)

        self._btn_missing = []
        for suit, name in enumerate(SUIT_NAMES):
            button = QPushButton(f"缺 {name}")
            button.setObjectName("missingBtn")
            button.clicked.connect(lambda _=False, s=suit: self._on_missing(s))
            layout.addWidget(button)
            self._btn_missing.append(button)

        layout.addSpacing(4)

        self._btn_hu = self._action_button("胡", "dangerBtn", self._on_hu)
        layout.addWidget(self._btn_hu)
        self._btn_pong = self._action_button("碰", "okBtn", self._on_pong)
        layout.addWidget(self._btn_pong)
        self._btn_exposed_kong = self._action_button(
            "直杠", "okBtn", self._on_exposed_kong
        )
        layout.addWidget(self._btn_exposed_kong)
        self._btn_concealed_kong = self._action_button(
            "暗杠", "okBtn", self._on_concealed_kong
        )
        layout.addWidget(self._btn_concealed_kong)
        self._btn_added_kong = self._action_button("碰杠", "okBtn", self._on_added_kong)
        layout.addWidget(self._btn_added_kong)
        self._btn_pass = self._action_button("过", "passBtn", self._on_pass)
        layout.addWidget(self._btn_pass)
        layout.addSpacing(8)
        self._btn_new_game = QPushButton("再来一局")
        self._btn_new_game.setObjectName("okBtn")
        self._btn_new_game.clicked.connect(lambda _=False: self._open_setup(None))
        layout.addWidget(self._btn_new_game)

        self._action_buttons = [
            self._btn_hu,
            self._btn_pong,
            self._btn_exposed_kong,
            self._btn_concealed_kong,
            self._btn_added_kong,
            self._btn_pass,
            self._btn_new_game,
        ]
        # Hidden until a match starts and _update_action_bar runs.
        for button in self._action_buttons:
            button.hide()
        for button in self._btn_missing:
            button.hide()
        self._bar_hint.setText("请先开始一局")
        return frame

    @staticmethod
    def _action_button(text: str, object_name: str | None, handler) -> QPushButton:
        button = QPushButton(text)
        if object_name:
            button.setObjectName(object_name)
        button.clicked.connect(handler)
        return button

    def _on_missing(self, suit: int) -> None:
        self._pending_kong = None
        if not self.controller.submit(bm.ACTION_CHOOSE_MISSING_OFFSET + suit):
            self._flash_hint("非法操作")

    def _on_hu(self) -> None:
        self._pending_kong = None
        if not self.controller.submit(bm.ACTION_HU):
            self._flash_hint("不能和牌")

    def _on_pong(self) -> None:
        self._pending_kong = None
        if not self.controller.submit(bm.ACTION_PONG):
            self._flash_hint("不能碰")

    def _on_exposed_kong(self) -> None:
        self._pending_kong = None
        if not self.controller.submit(bm.ACTION_EXPOSED_KONG):
            self._flash_hint("不能直杠")

    def _on_pass(self) -> None:
        self._pending_kong = None
        if not self.controller.submit(bm.ACTION_PASS):
            self._flash_hint("非法操作")

    def _on_concealed_kong(self) -> None:
        kinds = self._kong_kinds(bm.ACTION_CONCEALED_KONG_OFFSET)
        if not kinds:
            self._flash_hint("不能暗杠")
            return
        self._pending_kong = None
        if len(kinds) == 1:
            if not self.controller.submit(bm.ACTION_CONCEALED_KONG_OFFSET + kinds[0]):
                self._flash_hint("暗杠操作失败")
        else:
            self._pending_kong = "暗杠"
            self._refresh()

    def _on_added_kong(self) -> None:
        kinds = self._kong_kinds(bm.ACTION_ADDED_KONG_OFFSET)
        if not kinds:
            self._flash_hint("不能碰杠")
            return
        self._pending_kong = None
        if len(kinds) == 1:
            if not self.controller.submit(bm.ACTION_ADDED_KONG_OFFSET + kinds[0]):
                self._flash_hint("碰杠操作失败")
        else:
            self._pending_kong = "碰杠"
            self._refresh()

    def _kong_kinds(self, base: int) -> list[int]:
        return [
            action - base
            for action in self.controller.legal_action_ids()
            if base <= action < base + 27
        ]

    def _on_own_tile_clicked(
        self, kind: int, exchange_slot: tuple[int, int] | None = None
    ) -> None:
        if not self.controller.human_must_act():
            return
        phase = self.controller.game.phase
        if phase == bm.PHASE_EXCHANGE:
            previous_slots = self._exchange_selected_slots.copy()
            if exchange_slot is not None:
                if exchange_slot in self._exchange_selected_slots:
                    exchange_slot = next(
                        (
                            label.exchange_slot
                            for label in self._own_hand_labels
                            if label.kind == kind
                            and label.exchange_slot not in self._exchange_selected_slots
                        ),
                        exchange_slot,
                    )
                self._exchange_selected_slots.add(exchange_slot)
            if not self.controller.submit(bm.ACTION_EXCHANGE_TILE_OFFSET + kind):
                self._exchange_selected_slots = previous_slots
                self._flash_hint("非法操作")
            return
        if phase == bm.PHASE_TURN:
            if self._pending_kong:
                base = (
                    bm.ACTION_CONCEALED_KONG_OFFSET
                    if self._pending_kong == "暗杠"
                    else bm.ACTION_ADDED_KONG_OFFSET
                )
                if self.controller.submit(base + kind):
                    self._pending_kong = None
                else:
                    self._flash_hint(f"不能{self._pending_kong}这张牌")
                return
            if not self.controller.submit(bm.ACTION_DISCARD_OFFSET + kind):
                self._flash_hint("这张牌不能打出")
            return
        self._flash_hint("现在不能操作")

    def _flash_hint(self, text: str) -> None:
        self._bar_hint.setText(text)
        if self._pending_kong:
            self._pending_kong = None
            self._refresh()

    def _refresh(self) -> None:
        view = self.controller.view()
        self._view = view
        phase, actor = view["phase"], view["actor"]
        if phase not in (bm.PHASE_EXCHANGE, bm.PHASE_TURN) or actor != 0:
            self._pending_kong = None
        self._render_hands(view)
        self._render_seat_titles(view)
        self._render_discards(view)
        self._render_locked(view)
        self._render_melds(view)
        self._render_info(view)
        self._render_scores(view)
        self._update_action_bar(view)

    def _render_seat_titles(self, view: dict) -> None:
        actor = view["actor"]
        for rel in range(4):
            self._name_labels[rel].setText(self._seat_title(rel))
            active = actor == rel
            badge = self._badge_frames[rel]
            badge.setObjectName("seatBadgeActive" if active else "seatBadge")
            badge.style().unpolish(badge)
            badge.style().polish(badge)

    def _show_bar_tile(self, kind: int | None, text: str) -> None:
        if kind is None or kind < 0:
            self._bar_tile.hide()
            return
        if hasattr(self._bar_tile, "set_kind"):
            self._bar_tile.set_kind(kind, back=False)
        else:
            self._bar_tile.setText(kind_to_glyph(kind))
        self._bar_tile.setToolTip(kind_text(kind))
        self._bar_tile.show()
        if text:
            self._bar_hint.setText(text)

    def _update_action_bar(self, view: dict) -> None:
        """Show only legal contextual actions for the active phase."""
        for button in self._action_buttons:
            button.hide()
        for button in self._btn_missing:
            button.hide()
        self._bar_tile.hide()
        self._btn_hu.setText("胡")

        phase = view["phase"]
        if phase == bm.PHASE_FINISHED:
            self._bar_hint.setText("牌局结束")
            self._btn_new_game.show()
            return
        actor = view["actor"]
        if actor != 0:
            self._bar_hint.setText(f"等待 {SEAT_NAMES[actor]} 行动…")
            return

        legal = set(self.controller.legal_action_ids())
        if phase == bm.PHASE_EXCHANGE:
            picked = int(view["exchange_selection"].sum())
            self._bar_hint.setText(f"换三张: 点选第 {picked + 1}/3 张同花色牌")
        elif phase == bm.PHASE_CHOOSE_MISSING:
            self._bar_hint.setText("定缺: 选择要缺的一门")
            for suit in range(3):
                self._btn_missing[suit].setVisible(
                    bm.ACTION_CHOOSE_MISSING_OFFSET + suit in legal
                )
        elif phase == bm.PHASE_TURN:
            turn_hu_text = "自摸" if view["draw_tile"] >= 0 else "天胡"
            concealed_kinds = sorted(
                a - bm.ACTION_CONCEALED_KONG_OFFSET
                for a in legal
                if bm.ACTION_CONCEALED_KONG_OFFSET <= a < bm.ACTION_ADDED_KONG_OFFSET
            )
            added_kinds = sorted(
                a - bm.ACTION_ADDED_KONG_OFFSET
                for a in legal
                if bm.ACTION_ADDED_KONG_OFFSET <= a < bm.ACTION_PASS
            )
            if self._pending_kong:
                base = (
                    bm.ACTION_CONCEALED_KONG_OFFSET
                    if self._pending_kong == "暗杠"
                    else bm.ACTION_ADDED_KONG_OFFSET
                )
                kinds = sorted(a - base for a in legal if base <= a < base + 27)
                names = "、".join(kind_text(k) for k in kinds)
                self._bar_hint.setText(f"请点击要{self._pending_kong}的牌: {names}")
            else:
                hints = ["点牌打出"]
                if bm.ACTION_HU in legal:
                    hints.append(f"可{turn_hu_text}")
                if concealed_kinds:
                    hints.append(
                        "可暗杠 " + "、".join(kind_text(k) for k in concealed_kinds)
                    )
                if added_kinds:
                    hints.append(
                        "可碰杠 " + "、".join(kind_text(k) for k in added_kinds)
                    )
                self._bar_hint.setText(" · ".join(hints))
            self._btn_concealed_kong.setToolTip(
                "可暗杠: " + "、".join(kind_text(k) for k in concealed_kinds)
                if concealed_kinds
                else ""
            )
            self._btn_added_kong.setToolTip(
                "可碰杠: " + "、".join(kind_text(k) for k in added_kinds)
                if added_kinds
                else ""
            )
            if bm.ACTION_HU in legal:
                self._btn_hu.setText(turn_hu_text)
                self._btn_hu.show()
            self._btn_concealed_kong.setVisible(bool(concealed_kinds))
            self._btn_added_kong.setVisible(bool(added_kinds))
        elif phase == bm.PHASE_HU_RESPONSE:
            tile = int(view["response_tile"])
            self._show_bar_tile(
                tile,
                f"可和 {kind_text(tile)}" if tile >= 0 else "可和牌,或过",
            )
            self._btn_hu.setVisible(bm.ACTION_HU in legal)
            self._btn_pass.setVisible(bm.ACTION_PASS in legal)
        elif phase == bm.PHASE_MELD_RESPONSE:
            tile = int(view["response_tile"])
            self._show_bar_tile(
                tile,
                f"可碰/杠 {kind_text(tile)}" if tile >= 0 else "可碰/直杠,或过",
            )
            self._btn_pong.setVisible(bm.ACTION_PONG in legal)
            self._btn_exposed_kong.setVisible(bm.ACTION_EXPOSED_KONG in legal)
            self._btn_pass.setVisible(bm.ACTION_PASS in legal)

    def _render_hands(self, view: dict) -> None:
        own = count_to_hand(view["unlocked_hand"])
        row = self._hand_rows[0]
        self._clear_layout(row)
        self._own_hand_labels = []

        phase, actor = view["phase"], view["actor"]
        legal = set(self.controller.legal_action_ids()) if actor == 0 else set()
        kong_kinds: set[int] = set()
        kong_hint: dict[int, str] = {}
        if phase == bm.PHASE_EXCHANGE:
            clickable = {a for a in legal if a < 27}
            selected_counts = [int(value) for value in view["exchange_selection"]]
            selected_slots = self._sync_exchange_selected_slots(own, selected_counts)
            marker_kind = None
        elif phase == bm.PHASE_TURN:
            self._exchange_selected_slots.clear()
            selected_slots = set()
            if self._pending_kong:
                base = (
                    bm.ACTION_CONCEALED_KONG_OFFSET
                    if self._pending_kong == "暗杠"
                    else bm.ACTION_ADDED_KONG_OFFSET
                )
                clickable = {a - base for a in legal if base <= a < base + 27}
                kong_kinds = set(clickable)
                kong_hint = {
                    kind: f"选择{self._pending_kong} {kind_text(kind)}"
                    for kind in kong_kinds
                }
            else:
                clickable = {a - bm.ACTION_DISCARD_OFFSET for a in legal if a < 57}
                concealed_kinds = {
                    a - bm.ACTION_CONCEALED_KONG_OFFSET
                    for a in legal
                    if bm.ACTION_CONCEALED_KONG_OFFSET
                    <= a
                    < bm.ACTION_ADDED_KONG_OFFSET
                }
                added_kinds = {
                    a - bm.ACTION_ADDED_KONG_OFFSET
                    for a in legal
                    if bm.ACTION_ADDED_KONG_OFFSET <= a < bm.ACTION_PASS
                }
                kong_kinds = concealed_kinds | added_kinds
                kong_hint = {
                    kind: (
                        "可暗杠 " + kind_text(kind)
                        if kind in concealed_kinds
                        else "可碰杠 " + kind_text(kind)
                    )
                    for kind in kong_kinds
                }
            marker_kind = view["draw_tile"] if view["draw_tile"] >= 0 else None
        else:
            self._exchange_selected_slots.clear()
            selected_slots = set()
            clickable = set()
            marker_kind = None

        # Keep the drawn tile separated at the right edge, like a physical hand.
        if phase == bm.PHASE_TURN and marker_kind in own:
            own.remove(marker_kind)
            own.append(marker_kind)

        occurrences = [0] * 27
        row.addStretch(1)
        for index, kind in enumerate(own):
            exchange_slot = (kind, occurrences[kind])
            occurrences[kind] += 1
            selected = exchange_slot in selected_slots
            marker = bool(
                (
                    marker_kind is not None
                    and kind == marker_kind
                    and index == len(own) - 1
                )
                or kind in kong_kinds
            )
            tooltip = kong_hint.get(kind)
            label = tile_label(
                kind,
                size=_OWN_SIZE,
                font_pt=_OWN_FONT,
                selected=selected,
                clickable=kind in clickable,
                marker=marker,
                tooltip=tooltip,
            )
            label.exchange_slot = exchange_slot
            if kind in clickable:
                label.clicked.connect(
                    lambda clicked_kind, slot=exchange_slot: self._on_own_tile_clicked(
                        clicked_kind, slot
                    )
                )
            if marker:
                row.addSpacing(8)
            row.addWidget(label)
            self._own_hand_labels.append(label)
        row.addStretch(1)

        for rel in (1, 2, 3):
            hand = self._hand_rows[rel]
            self._clear_layout(hand)
            count = view["unlocked_hand_counts"][rel]
            labels = []
            for _ in range(count):
                size = _AI_SIZE if rel == 2 else (_AI_SIZE[1], _AI_SIZE[0])
                label = tile_label(0, back=True, size=size, font_pt=_AI_FONT)
                index = len(labels)
                if rel == 2:
                    hand.addWidget(label)
                else:
                    # A bloodflow hand can contain more than 13 tiles after a win.
                    # Keep the physical wall compact by adding a new outer column.
                    hand.addWidget(label, index % 13, index // 13)
                labels.append(label)
            self._ai_hand_labels[rel] = labels

    def _sync_exchange_selected_slots(
        self, hand: list[int], selected_counts: list[int]
    ) -> set[tuple[int, int]]:
        slots_by_kind: list[list[tuple[int, int]]] = [[] for _ in range(27)]
        occurrences = [0] * 27
        for kind in hand:
            slot = (kind, occurrences[kind])
            occurrences[kind] += 1
            slots_by_kind[kind].append(slot)

        selected: set[tuple[int, int]] = set()
        for kind, count in enumerate(selected_counts):
            slots = slots_by_kind[kind]
            retained = [slot for slot in slots if slot in self._exchange_selected_slots]
            selected.update(retained[:count])
            needed = count - min(count, len(retained))
            for slot in slots:
                if needed == 0:
                    break
                if slot in selected:
                    continue
                selected.add(slot)
                needed -= 1

        self._exchange_selected_slots = selected
        return selected

    def _render_locked(self, view: dict) -> None:
        """Render winning structures separately from the active concealed hand."""
        for rel in range(4):
            layout = self._locked_layouts[rel].layout()
            self._clear_layout(layout)
            self._locked_labels[rel] = []
            kinds = count_to_hand(view["locked"][rel])
            legal = (
                set(self.controller.legal_action_ids())
                if rel == 0 and view["actor"] == 0
                else set()
            )
            added_kinds = {
                action - bm.ACTION_ADDED_KONG_OFFSET
                for action in legal
                if bm.ACTION_ADDED_KONG_OFFSET <= action < bm.ACTION_PASS
            }
            pending_added = self._pending_kong == "碰杠"
            selectable = added_kinds if pending_added else set()
            for index, kind in enumerate(kinds):
                if rel in (0, 2):
                    row, column = divmod(index, 8)
                    size = _LOCKED_SIZE
                else:
                    row, column = index % 10, index // 10
                    size = (_LOCKED_SIZE[1], _LOCKED_SIZE[0])
                tooltip = f"和牌锁定 · {kind_text(kind)}"
                if kind in added_kinds:
                    tooltip = (
                        f"选择碰杠 {kind_text(kind)}"
                        if pending_added
                        else f"可碰杠 {kind_text(kind)}"
                    )
                label = tile_label(
                    kind,
                    size=size,
                    font_pt=_LOCKED_FONT,
                    marker=True,
                    clickable=kind in selectable,
                    tooltip=tooltip,
                )
                if kind in selectable:
                    label.clicked.connect(self._on_own_tile_clicked)
                self._locked_labels[rel].append(label)
                layout.addWidget(label, row, column)

            if not kinds:
                placeholder = QLabel("—")
                placeholder.setObjectName("seatInfo")
                layout.addWidget(placeholder, 0, 0)

    def _render_melds(self, view: dict) -> None:
        """Render each open meld as a stable visual group."""
        melds = view["melds"]
        for rel in range(4):
            layout = self._meld_layouts[rel].layout()
            self._clear_layout(layout)
            meld_index = 0
            for slot in range(4):
                tile = int(melds[rel][slot][0])
                if tile == 255:
                    continue
                kind = int(melds[rel][slot][1])
                group = QWidget()
                group.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                group_layout = (
                    QVBoxLayout(group) if rel in (1, 3) else QHBoxLayout(group)
                )
                group_layout.setContentsMargins(0, 0, 0, 0)
                group_layout.setSpacing(2)
                tag = QLabel(MELD_NAMES[kind])
                tag.setObjectName("meldTag")
                tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
                tag.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                group_layout.addWidget(tag)
                copies = 4 if kind in (1, 2, 3) else 3
                tile_size = (
                    _PUBLIC_SIZE
                    if rel in (0, 2)
                    else (
                        _PUBLIC_SIZE[1],
                        _PUBLIC_SIZE[0],
                    )
                )
                for _ in range(copies):
                    group_layout.addWidget(
                        tile_label(tile, size=tile_size, font_pt=_PUBLIC_FONT)
                    )

                if rel in (0, 2):
                    layout.addWidget(group, meld_index // 2, (meld_index % 2) * 3, 1, 3)
                else:
                    layout.addWidget(group, 0, meld_index)
                meld_index += 1

            if meld_index == 0:
                placeholder = QLabel("—")
                placeholder.setObjectName("seatInfo")
                layout.addWidget(placeholder, 0, 0)

    def _render_discards(self, view: dict) -> None:
        rivers = [[] for _ in range(4)]
        for tile, owner in view["river"]:
            tile = int(tile)
            owner = int(owner)
            if tile == 255 or owner == 255:
                continue
            if 0 <= owner < 4:
                rivers[owner].append(tile)

        for rel in range(4):
            layout = self._discard_layouts[rel].layout()
            self._clear_layout(layout)
            kinds = rivers[rel]
            if not kinds:
                placeholder = QLabel("—")
                placeholder.setObjectName("seatInfo")
                layout.addWidget(placeholder, 0, 0)
                continue

            for index, kind in enumerate(kinds):
                if rel in (0, 2):
                    row, column = divmod(index, 6)
                    size = _PUBLIC_SIZE
                else:
                    row, column = index % 6, index // 6
                    size = (_PUBLIC_SIZE[1], _PUBLIC_SIZE[0])
                layout.addWidget(
                    tile_label(kind, size=size, font_pt=_PUBLIC_FONT), row, column
                )

    def _render_info(self, view: dict) -> None:
        phase = view["phase"]
        self._phase_title.setText(PHASE_NAMES[phase])
        self._remain_value.setText(str(view["wall_remaining"]))

        dealer = view["dealer"]
        direction = DIRECTION_NAMES.get(view["direction"], "—")
        self._center_meta.setText(
            f"庄 {WIND_NAMES[self._abs(dealer)]} · 换牌 {direction}"
        )

        position_map = {
            "south": 0,
            "east": 1,
            "north": 2,
            "west": 3,
        }
        for key, rel in position_map.items():
            label = self._wind_labels[key]
            label.setText(WIND_NAMES[self._abs(rel)])
            if rel == dealer:
                label.setObjectName("windActive")
            else:
                label.setObjectName("windLabel")
            # Force style refresh after objectName change.
            label.style().unpolish(label)
            label.style().polish(label)

    def _render_scores(self, view: dict) -> None:
        scores = view["scores"]
        best = max(scores)
        for rel, label in enumerate(self._score_labels):
            text = f"{SEAT_NAMES[rel]} · {WIND_NAMES[self._abs(rel)]} · {scores[rel]}"
            if scores[rel] == best:
                text += "  ← 领先"
                label.setObjectName("sideTextLead")
            else:
                label.setObjectName("sideText")
            label.setText(text)
            label.style().unpolish(label)
            label.style().polish(label)

    def _rel(self, seat: int) -> int:
        """Absolute seat -> relative to human (0=self,1=right,2=opposite,3=left)."""
        return (seat - self._human_seat) & 3

    def _abs(self, relative_seat: int) -> int:
        """Relative seat -> absolute engine seat."""
        return (self._human_seat + relative_seat) & 3

    def _seat_title(self, seat: int) -> str:
        view = getattr(self, "_view", None)
        rel = seat
        abs_seat = self._abs(rel)
        parts = [f"{SEAT_NAMES[rel]}({WIND_NAMES[abs_seat]})"]
        if view is not None:
            parts.append(f"{view['scores'][rel]} 分")
            missing = view["missing"][rel]
            if missing >= 0:
                parts.append(f"缺:{SUIT_NAMES[missing]}")
            if view["has_won"][rel]:
                parts.append("已和")
            if rel == view["dealer"]:
                parts.append("庄")
        return " · ".join(parts)

    def _append_log(self, lines: list[str]) -> None:
        for line in lines:
            self._log.appendPlainText(line)

    def _on_finished(self, rankings: tuple, scores: tuple) -> None:
        order = " > ".join(SEAT_NAMES[self._rel(r)] for r in rankings)
        self._append_log([f"对局结束 · 排名: {order} · 分数: {list(scores)}"])
        self.statusBar().showMessage(
            f"对局结束,冠军: {SEAT_NAMES[self._rel(rankings[0])]}"
        )

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                MainWindow._clear_layout(item.layout())

    def own_hand_label_count(self) -> int:
        return len(self._own_hand_labels)

    def ai_hand_label_counts(self) -> dict[int, int]:
        return {
            seat: len(labels)
            for seat, labels in self._ai_hand_labels.items()
            if seat != 0
        }

    def log_text(self) -> str:
        return self._log.toPlainText()

    def info_text(self) -> str:
        wall = f"{self._remain_value.text()} 张"
        return f"{self._phase_title.text()} | {wall} | {self._center_meta.text()}"
