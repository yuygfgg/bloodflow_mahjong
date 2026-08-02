from __future__ import annotations

SUIT_NAMES = ("万", "条", "筒")
SUIT_GLYPH_BASES = (0x1F007, 0x1F010, 0x1F019)
BACK_GLYPH = "\U0001f02b"
TILE_KIND_COUNT = 27
MELD_NAMES = ("碰", "直杠", "碰杠", "暗杠")
PHASE_NAMES = ("换三张", "定缺", "行牌", "和牌响应", "碰杠响应", "终局")
SEAT_NAMES = ("你", "下家", "对家", "上家")
DIRECTION_NAMES = {1: "左", 2: "对家", 3: "右"}


def kind_suit(kind: int) -> int:
    return kind // 9


def kind_rank(kind: int) -> int:
    return kind % 9


def kind_to_glyph(kind: int) -> str:
    return chr(SUIT_GLYPH_BASES[kind_suit(kind)] + kind_rank(kind))


def kind_text(kind: int) -> str:
    return f"{kind_rank(kind) + 1}{SUIT_NAMES[kind_suit(kind)]}"


def count_to_hand(counts) -> list[int]:
    """把 27 维张数数组展开为升序的牌种列表(每个 kind 重复 count 次)。"""
    return [kind for kind, n in enumerate(counts) if n for _ in range(n)]
