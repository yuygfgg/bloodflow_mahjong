from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QLabel, QSizePolicy

from ..model.tiles import BACK_GLYPH, kind_text, kind_to_glyph

_FONT_CANDIDATES = (
    "Noto Sans Symbols 2",
    "Symbola",
    "Segoe UI Symbol",
    "Apple Color Emoji",
    "Noto Color Emoji",
)

_resolved_family: str | None = None
_resolved_ok = False

# Face-up ivory tile, amber back, and restrained state borders.
_FACE_UP_QSS = (
    "background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
    " stop:0 #fffdf7, stop:0.55 #f7f0de, stop:1 #ebe2ca);"
    " color: #1f1a14;"
    " border: 1px solid #c9b27a;"
    " border-top: 1px solid #f4e8c4;"
    " border-left: 1px solid #f4e8c4;"
    " border-radius: 6px;"
)
_BACK_QSS = (
    "background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
    " stop:0 #f1c350, stop:0.18 #d99a25, stop:1 #b96d14);"
    " color: #fff2c8;"
    " border: 1px solid #75440f;"
    " border-top: 2px solid #ffe092;"
    " border-radius: 5px;"
)
_BORDER_SELECTED = " border: 2px solid #e53935;"
_BORDER_CLICKABLE = " border: 1px solid #8fae9c;"
_BORDER_MARKER = " border: 2px solid #f9a825;"


def resolve_family(glyph: str = "\U0001f007") -> str | None:
    """Return the preferred font family that can render mahjong glyphs, or None."""
    global _resolved_family, _resolved_ok
    if not _resolved_ok:
        probe = QFont()
        for family in _FONT_CANDIDATES:
            font = QFont(family, probe.pointSize() or 12)
            if QFontMetrics(font).inFont(glyph):
                _resolved_family = family
                break
        _resolved_ok = True
    return _resolved_family


class ClickableTileLabel(QLabel):
    """Clickable mahjong tile that emits clicked(kind)."""

    clicked = Signal(int)

    def __init__(self, kind: int) -> None:
        super().__init__()
        self.kind = kind

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.kind)
        super().mousePressEvent(event)

    def set_kind(self, kind: int, *, back: bool | None = None) -> None:
        """Update face glyph. ``back`` is accepted for API symmetry with the factory."""
        del back
        self.kind = kind
        family = resolve_family()
        self.setText(kind_to_glyph(kind) if family else kind_text(kind))
        self.setToolTip(kind_text(kind))


def tile_label(
    kind: int,
    *,
    back: bool = False,
    size: tuple[int, int] = (38, 50),
    font_pt: int = 20,
    selected: bool = False,
    clickable: bool = False,
    marker: bool = False,
    tooltip: str | None = None,
) -> QLabel:
    """Build a mahjong tile QLabel. When back=True, kind is unused."""
    family = resolve_family()
    if back:
        text = BACK_GLYPH
        fallback = "牌"
    else:
        text = kind_to_glyph(kind)
        fallback = kind_text(kind)

    label = ClickableTileLabel(kind) if clickable else QLabel()
    label.kind = kind  # type: ignore[attr-defined]
    label.setText(text if family else fallback)
    font = QFont(family or "", font_pt)
    label.setFont(font)
    label.setFixedSize(*size)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    label.setProperty("selected", selected)

    qss = _BACK_QSS if back else _FACE_UP_QSS
    if selected:
        qss += _BORDER_SELECTED
    elif marker:
        qss += _BORDER_MARKER
    elif clickable:
        qss += _BORDER_CLICKABLE
    label.setStyleSheet(qss)

    if clickable and not back:
        label.setCursor(Qt.CursorShape.PointingHandCursor)
    if tooltip:
        label.setToolTip(tooltip)
    elif not back:
        label.setToolTip(kind_text(kind))
    return label
