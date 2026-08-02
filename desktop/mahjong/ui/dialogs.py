from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from ..model.policy import DIFFICULTIES
from ..model.tiles import SEAT_NAMES

WIND_NAMES = ("东", "南", "西", "北")

_DIALOG_QSS = """
QDialog {
    background: #142820;
    color: #e8f0ea;
}
QLabel {
    color: #dce8e0;
    font-size: 13px;
}
QRadioButton {
    color: #e8f0ea;
    spacing: 6px;
}
QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border-radius: 7px;
    border: 1px solid #c4a35a;
    background: #0f1f19;
}
QRadioButton::indicator:checked {
    background: #c4a35a;
}
QComboBox, QLineEdit {
    background: #0f1f19;
    color: #f3efe4;
    border: 1px solid #6d7d74;
    border-radius: 8px;
    padding: 6px 10px;
    min-height: 28px;
}
QComboBox:hover, QLineEdit:hover {
    border-color: #c4a35a;
}
QComboBox QAbstractItemView {
    background: #0f1f19;
    color: #f3efe4;
    selection-background-color: #2e7d32;
}
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #3a4a44, stop:1 #24332e);
    color: #f2efe6;
    border: 1px solid #6d7d74;
    border-radius: 14px;
    padding: 8px 18px;
    font-size: 14px;
    font-weight: 700;
    min-width: 88px;
}
QPushButton:hover {
    border-color: #c4a35a;
    background: #2f413a;
}
QPushButton#startBtn {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #43a047, stop:1 #2e7d32);
    border: 1px solid #81c784;
    color: white;
}
QPushButton#startBtn:hover {
    background: #388e3c;
}
"""


class SetupDialog(QDialog):
    """New-game setup: seat (dealer/east is seat 0), AI strength, seed."""

    def __init__(
        self,
        parent=None,
        defaults: dict | None = None,
        *,
        required: bool = False,
    ) -> None:
        super().__init__(parent)
        self._required = required
        self.setWindowTitle("新牌局")
        self.setModal(True)
        if required:
            self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self.setMinimumWidth(420)
        self.setStyleSheet(_DIALOG_QSS)
        self._defaults = defaults or {}

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)
        form = QFormLayout()
        form.setSpacing(12)
        form.setHorizontalSpacing(14)

        seat_row = QHBoxLayout()
        self._seat_group = QButtonGroup(self)
        for seat in range(4):
            text = f"{WIND_NAMES[seat]}家"
            if seat == 0:
                text += " (庄家)"
            radio = QRadioButton(text)
            self._seat_group.addButton(radio, seat)
            seat_row.addWidget(radio)
            if seat == self._defaults.get("seat", 0):
                radio.setChecked(True)
        form.addRow("你的座位:", seat_row)

        self._ai_labels: dict[int, QLabel] = {}
        self._ai_combos: dict[int, QComboBox] = {}
        default_difficulties = self._defaults.get("difficulties", {})
        seat = self._defaults.get("seat", 0)
        for rel in (1, 2, 3):
            abs_seat = (seat + rel) & 3
            label = QLabel()
            combo = QComboBox()
            combo.addItems(DIFFICULTIES)
            combo.setCurrentText(default_difficulties.get(abs_seat, "标准"))
            form.addRow(label, combo)
            self._ai_labels[rel] = label
            self._ai_combos[rel] = combo

        self._seed_edit = QLineEdit()
        self._seed_edit.setPlaceholderText("留空则随机")
        self._seed_edit.setText(str(self._defaults.get("seed", "") or ""))
        form.addRow("随机种子:", self._seed_edit)
        root.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self._cancel_button = QPushButton("取消")
        self._cancel_button.clicked.connect(self.reject)
        self._cancel_button.setVisible(not required)
        start = QPushButton("开始")
        start.setObjectName("startBtn")
        start.setDefault(True)
        start.clicked.connect(self.accept)
        buttons.addWidget(self._cancel_button)
        buttons.addWidget(start)
        root.addLayout(buttons)

        self._seat_group.idToggled.connect(self._update_labels)
        self._update_labels()

    def reject(self) -> None:
        """Keep a required startup dialog open until setup is accepted."""
        if self._required:
            return
        super().reject()

    def closeEvent(self, event) -> None:
        if self._required:
            event.ignore()
            return
        super().closeEvent(event)

    def _update_labels(self) -> None:
        seat = self._seat_group.checkedId()
        if seat < 0:
            return
        for rel in (1, 2, 3):
            abs_seat = (seat + rel) & 3
            self._ai_labels[rel].setText(
                f"{SEAT_NAMES[rel]} ({WIND_NAMES[abs_seat]}家) 强度:"
            )

    def values(self) -> dict:
        seat = self._seat_group.checkedId()
        difficulties = {
            (seat + rel) & 3: self._ai_combos[rel].currentText() for rel in (1, 2, 3)
        }
        text = self._seed_edit.text().strip()
        try:
            seed = int(text) if text else None
        except ValueError:
            seed = None
        return {"seat": seat, "difficulties": difficulties, "seed": seed}
