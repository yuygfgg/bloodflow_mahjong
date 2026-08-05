#!/usr/bin/env python3
"""Convert OpenRiichi source assets into the Blood Flow web client bundle."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Engine tile index = suit * 9 + rank, suit 0=Characters 1=Bamboo 2=Dots.
# OpenRiichi names the same three suits Man / Sou / Pin.
SUIT_PREFIX = ("Man", "Sou", "Pin")
SUIT_LABEL = ("万", "条", "筒")
TILE_COUNT = 27

CELL_PX = 128
GUTTER_PX = 8
CELL_PITCH = CELL_PX + 2 * GUTTER_PX  # 144
ATLAS_COLS = 8
ATLAS_ROWS = 4
ATLAS_W = ATLAS_COLS * CELL_PITCH  # 1152
ATLAS_H = ATLAS_ROWS * CELL_PITCH  # 576
BLANK_CELL = 27
BACK_CELL = 28
WHITE_CELL = 29
# Options.vala:19 -- tile_back_color = Color(0, 0.5f, 1, 1)
TILE_BACK_RGB = (0, 128, 255)

# Neutral table sounds only; the spoken Japanese calls do not apply here.
SOUNDS = ("tile", "draw", "discard", "slide", "flip", "reveal",
          "click", "mouse_over", "score_count", "hint", "fade_in")


# --------------------------------------------------------------------------- #
# Wavefront OBJ
# --------------------------------------------------------------------------- #


@dataclass
class Primitive:
    name: str
    positions: list[float] = field(default_factory=list)
    normals: list[float] = field(default_factory=list)
    uvs: list[float] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    lookup: dict = field(default_factory=dict, repr=False)

    @property
    def vertex_count(self) -> int:
        return len(self.positions) // 3

    def emit(self, key, position, uv, normal) -> None:
        index = self.lookup.get(key)
        if index is None:
            index = self.vertex_count
            self.lookup[key] = index
            self.positions.extend(position)
            self.uvs.extend(uv)
            self.normals.extend(normal)
        self.indices.append(index)


def _corner(token: str, nv: int, nt: int, nn: int):
    raw = (token.split("/") + ["", ""])[:3]

    def resolve(text: str, count: int):
        if not text:
            return None
        value = int(text)
        return value - 1 if value > 0 else count + value

    return resolve(raw[0], nv), resolve(raw[1], nt), resolve(raw[2], nn)


def parse_obj(path: Path) -> list[Primitive]:
    """Parse a Blender-exported OBJ into one Primitive per ``o`` group.

    Quads and larger polygons are fan-triangulated. OBJ's bottom-left UV origin
    is flipped to glTF's top-left so textures load with ``flipY = false``.
    """
    positions: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []
    prims: list[Primitive] = []
    current: Primitive | None = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]
        if tag == "v":
            positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif tag == "vt":
            positions_u = float(parts[1])
            positions_v = float(parts[2]) if len(parts) > 2 else 0.0
            uvs.append((positions_u, 1.0 - positions_v))
        elif tag == "vn":
            normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif tag in ("o", "g"):
            current = Primitive(name=" ".join(parts[1:]) or f"mesh{len(prims)}")
            prims.append(current)
        elif tag == "f":
            if current is None:
                current = Primitive(name="mesh0")
                prims.append(current)
            corners = [
                _corner(tok, len(positions), len(uvs), len(normals))
                for tok in parts[1:]
            ]
            for i in range(1, len(corners) - 1):
                for vi, ti, ni in (corners[0], corners[i], corners[i + 1]):
                    current.emit(
                        (vi, ti, ni),
                        positions[vi] if vi is not None else (0.0, 0.0, 0.0),
                        uvs[ti] if ti is not None else (0.0, 0.0),
                        normals[ni] if ni is not None else (0.0, 1.0, 0.0),
                    )

    return [p for p in prims if p.indices]


# --------------------------------------------------------------------------- #
# glTF 2.0 / GLB writer
# --------------------------------------------------------------------------- #

COMP_FLOAT = 5126
COMP_USHORT = 5123
COMP_UINT = 5125


class GlbBuilder:
    """Single-buffer GLB writer."""

    def __init__(self) -> None:
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []
        self.meshes: list[dict] = []
        self.nodes: list[dict] = []

    def _view(self, payload: bytes) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        offset = len(self.blob)
        self.blob.extend(payload)
        self.views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        )
        return len(self.views) - 1

    def vec(self, values: list[float], stride: int) -> int:
        payload = struct.pack(f"<{len(values)}f", *values)
        view = self._view(payload)
        count = len(values) // stride
        acc: dict = {
            "bufferView": view,
            "componentType": COMP_FLOAT,
            "count": count,
            "type": {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}[stride],
        }
        if stride == 3:
            cols = [values[i::3] for i in range(3)]
            acc["min"] = [min(c) for c in cols]
            acc["max"] = [max(c) for c in cols]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def idx(self, values: list[int]) -> int:
        wide = max(values, default=0) > 0xFFFF
        fmt = "I" if wide else "H"
        payload = struct.pack(f"<{len(values)}{fmt}", *values)
        self.accessors.append(
            {
                "bufferView": self._view(payload),
                "componentType": COMP_UINT if wide else COMP_USHORT,
                "count": len(values),
                "type": "SCALAR",
            }
        )
        return len(self.accessors) - 1


    def add_mesh(self, name: str, prims: list[dict]) -> int:
        self.meshes.append({"name": name, "primitives": prims})
        self.nodes.append({"name": name, "mesh": len(self.meshes) - 1})
        return len(self.nodes) - 1

    def primitive(
        self,
        prim: Primitive,
        material: int | None = None,
        extra: dict[str, list[float]] | None = None,
    ) -> dict:
        attributes = {
            "POSITION": self.vec(prim.positions, 3),
            "NORMAL": self.vec(prim.normals, 3),
            "TEXCOORD_0": self.vec(prim.uvs, 2),
        }
        for key, values in (extra or {}).items():
            attributes[key] = self.vec(values, 1)
        out: dict = {
            "attributes": attributes,
            "indices": self.idx(prim.indices),
            "mode": 4,
        }
        if material is not None:
            out["material"] = material
        return out

    def write(self, path: Path, materials: list[dict], notice: str) -> None:
        gltf = {
            "asset": {"version": "2.0", "generator": "bloodflow extract_assets.py",
                      "copyright": notice},
            "scene": 0,
            "scenes": [{"nodes": list(range(len(self.nodes)))}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "accessors": self.accessors,
            "bufferViews": self.views,
            "buffers": [{"byteLength": len(self.blob)}],
        }
        if materials:
            gltf["materials"] = materials

        json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        json_chunk += b" " * (-len(json_chunk) % 4)
        bin_chunk = bytes(self.blob)
        bin_chunk += b"\0" * (-len(bin_chunk) % 4)

        total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.write(struct.pack("<4sII", b"glTF", 2, total))
            handle.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
            handle.write(json_chunk)
            handle.write(struct.pack("<I4s", len(bin_chunk), b"BIN\0"))
            handle.write(bin_chunk)


# --------------------------------------------------------------------------- #
# Tile atlas
# --------------------------------------------------------------------------- #


def cell_rect(cell: int) -> tuple[int, int]:
    """Top-left pixel of a cell's 128x128 content box."""
    col, row = cell % ATLAS_COLS, cell // ATLAS_COLS
    return col * CELL_PITCH + GUTTER_PX, row * CELL_PITCH + GUTTER_PX


def paste_with_gutter(atlas: Image.Image, cell: int, tile: Image.Image) -> None:
    """Paste one 128x128 face and clamp-extend its edges into the gutter.

    The gutter keeps mip levels from bleeding neighbouring faces into each
    other; clamping (rather than transparent padding) matches how OpenRiichi's
    per-tile textures behave at their own borders.
    """
    if tile.size != (CELL_PX, CELL_PX):
        tile = tile.resize((CELL_PX, CELL_PX), Image.LANCZOS)
    x0, y0 = cell_rect(cell)
    g = GUTTER_PX
    # Centre, then the four edge strips, then the four corners.
    atlas.paste(tile, (x0, y0))
    atlas.paste(tile.crop((0, 0, CELL_PX, 1)).resize((CELL_PX, g)), (x0, y0 - g))
    atlas.paste(
        tile.crop((0, CELL_PX - 1, CELL_PX, CELL_PX)).resize((CELL_PX, g)),
        (x0, y0 + CELL_PX),
    )
    atlas.paste(tile.crop((0, 0, 1, CELL_PX)).resize((g, CELL_PX)), (x0 - g, y0))
    atlas.paste(
        tile.crop((CELL_PX - 1, 0, CELL_PX, CELL_PX)).resize((g, CELL_PX)),
        (x0 + CELL_PX, y0),
    )
    for cx, cy, sx, sy in (
        (x0 - g, y0 - g, 0, 0),
        (x0 + CELL_PX, y0 - g, CELL_PX - 1, 0),
        (x0 - g, y0 + CELL_PX, 0, CELL_PX - 1),
        (x0 + CELL_PX, y0 + CELL_PX, CELL_PX - 1, CELL_PX - 1),
    ):
        atlas.paste(tile.getpixel((sx, sy)), (cx, cy, cx + g, cy + g))


def build_tile_atlas(textures: Path, out_dir: Path) -> dict:
    """Pack the 27 suited faces plus blank / back / white into one atlas."""
    src = textures / "Tiles" / "Regular"
    atlas = Image.new("RGBA", (ATLAS_W, ATLAS_H), (0, 0, 0, 0))

    entries: list[dict] = []
    for tile in range(TILE_COUNT):
        suit, rank = divmod(tile, 9)
        name = f"{SUIT_PREFIX[suit]}{rank + 1}"
        face = Image.open(src / f"{name}.png").convert("RGBA")
        # OpenRiichi draws the transparent glyph texture over the tile's
        # opaque white front material. Flatten that same composition into the
        # atlas because the browser uses one shared material for the merged
        # tile mesh. Keeping the alpha channel here would render transparent
        # pixels as the dark table background in Three.js.
        white = Image.new("RGBA", face.size, (255, 255, 255, 255))
        face = Image.alpha_composite(white, face)
        paste_with_gutter(atlas, tile, face)
        entries.append(
            {
                "tile": tile,
                "cell": tile,
                "suit": suit,
                "rank": rank,
                "label": f"{rank + 1}{SUIT_LABEL[suit]}",
                "source": name,
            }
        )

    paste_with_gutter(atlas, BLANK_CELL,
                      Image.open(src / "Blank.png").convert("RGBA"))
    paste_with_gutter(
        atlas, BACK_CELL,
        Image.new("RGBA", (CELL_PX, CELL_PX), (*TILE_BACK_RGB, 255)),
    )
    paste_with_gutter(
        atlas, WHITE_CELL, Image.new("RGBA", (CELL_PX, CELL_PX), (255, 255, 255, 255))
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    atlas.save(out_dir / "tiles.webp", "WEBP", lossless=True, method=6)

    return {
        "image": "textures/tiles.webp",
        "width": ATLAS_W,
        "height": ATLAS_H,
        "cell": CELL_PX,
        "gutter": GUTTER_PX,
        "pitch": CELL_PITCH,
        "columns": ATLAS_COLS,
        "rows": ATLAS_ROWS,
        # Per-instance UV offset step; cell N adds (N%8*du, N//8*dv).
        "uvStep": [CELL_PITCH / ATLAS_W, CELL_PITCH / ATLAS_H],
        "blankCell": BLANK_CELL,
        "backCell": BACK_CELL,
        "whiteCell": WHITE_CELL,
        "backColor": [c / 255 for c in TILE_BACK_RGB],
        "tiles": entries,
    }


# --------------------------------------------------------------------------- #
# Meshes
# --------------------------------------------------------------------------- #


def bounds_of(prim: Primitive) -> dict:
    cols = [prim.positions[i::3] for i in range(3)]
    lo = [min(c) for c in cols]
    hi = [max(c) for c in cols]
    return {"min": lo, "max": hi, "size": [hi[i] - lo[i] for i in range(3)]}


def build_tile_glb(models: Path, out_dir: Path, quality: str, notice: str) -> dict:
    """Emit one merged tile mesh whose UVs are pre-baked into atlas cell 0.

    OpenRiichi keeps the face ("Top") and back ("Bottom") as separate objects
    with separate materials, the back untextured and flat-coloured. The web
    client instead needs a single shared material, so the back's degenerate UV
    is retargeted at the atlas' solid back-colour cell and a ``_FACE`` flag
    marks which vertices may be shifted by the per-instance cell offset.
    """
    prims = parse_obj(models / f"tile_{quality}.obj")
    top = next(p for p in prims if p.name.startswith("Top"))
    bottom = next(p for p in prims if p.name.startswith("Bottom"))

    fx0, fy0 = cell_rect(0)
    bx0, by0 = cell_rect(BACK_CELL)
    back_u = (bx0 + CELL_PX / 2) / ATLAS_W
    back_v = (by0 + CELL_PX / 2) / ATLAS_H

    merged = Primitive(name=f"tile_{quality}")
    faces: list[float] = []
    for prim, is_front in ((top, True), (bottom, False)):
        offset = merged.vertex_count
        merged.positions.extend(prim.positions)
        merged.normals.extend(prim.normals)
        for i in range(prim.vertex_count):
            u, v = prim.uvs[2 * i], prim.uvs[2 * i + 1]
            if is_front:
                merged.uvs.extend(
                    ((fx0 + u * CELL_PX) / ATLAS_W, (fy0 + v * CELL_PX) / ATLAS_H)
                )
            else:
                merged.uvs.extend((back_u, back_v))
            faces.append(1.0 if is_front else 0.0)
        merged.indices.extend(i + offset for i in prim.indices)

    glb = GlbBuilder()
    glb.add_mesh(f"tile_{quality}", [glb.primitive(merged, extra={"_FACE": faces})])
    glb.write(out_dir / f"tile_{quality}.glb", [], notice)

    tb, bb = bounds_of(top), bounds_of(bottom)
    # Cell-local sub-rect the face geometry actually samples. The source art has
    # transparent padding on the right, so the model stops short of u=1; 2D HUD
    # sprites must crop to the same rect or they will not match the 3D faces.
    fu = top.uvs[0::2]
    fv = top.uvs[1::2]
    return {
        "file": f"models/tile_{quality}.glb",
        "vertices": merged.vertex_count,
        "triangles": len(merged.indices) // 3,
        # RenderTile.vala:33 -- obb = (front.x, front.y + back.y, front.z)
        "obb": [tb["size"][0], tb["size"][1] + bb["size"][1], tb["size"][2]],
        "front": tb,
        "back": bb,
        "faceUv": {
            "min": [min(fu), min(fv)],
            "max": [max(fu), max(fv)],
        },
    }


def build_model_glb(models: Path, out_dir: Path, name: str, notice: str) -> dict:
    """Convert one OBJ to a geometry-only GLB.

    Materials stay out of the GLB on purpose: the client assigns them in code so
    quality tiers and custom felt textures can be swapped without re-exporting.
    """
    prims = parse_obj(models / f"{name}.obj")
    glb = GlbBuilder()
    glb.add_mesh(name, [glb.primitive(p) for p in prims])
    glb.write(out_dir / f"{name}.glb", [], notice)

    cols = [
        [v for p in prims for v in p.positions[i::3]] for i in range(3)
    ]
    lo = [min(c) for c in cols]
    hi = [max(c) for c in cols]
    return {
        "file": f"models/{name}.glb",
        "objects": [p.name for p in prims],
        "vertices": sum(p.vertex_count for p in prims),
        "triangles": sum(len(p.indices) // 3 for p in prims),
        "min": lo,
        "max": hi,
        "size": [hi[i] - lo[i] for i in range(3)],
    }


# --------------------------------------------------------------------------- #
# Textures, audio, fonts
# --------------------------------------------------------------------------- #

TEXTURES = {
    "table_high": "table_high.png",
    "table_low": "table_low.png",
    "field_high": "field_high.png",
    "field_low": "field_low.png",
    "field_marble": "field_marble.png",
    "table_center": "table_center.png",
    "stick_100": "Sticks/Stick100.png",
    "stick_1000": "Sticks/Stick1000.png",
    "button": "Buttons/MenuButton.png",
    "button_small": "Buttons/MenuButtonSmall.png",
    "button_big": "Buttons/MenuButtonBig.png",
    "score_background": "Menu/score_background.png",
}


def convert_textures(textures: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    for key, rel in TEXTURES.items():
        image = Image.open(textures / rel)
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        image.save(out_dir / f"{key}.webp", "WEBP", quality=92, method=6)
        manifest[key] = {
            "file": f"textures/{key}.webp",
            "width": image.width,
            "height": image.height,
        }
    return manifest


def convert_audio(audio: Path, out_dir: Path, music: bool) -> dict:
    """Re-encode the neutral table sounds to Ogg Opus.

    ``pon`` / ``kan`` / ``ron`` / ``tsumo`` / ``chii`` / ``riichi`` are spoken
    Japanese calls and are intentionally left behind: Blood Flow announces
    胡 / 碰 / 杠 / 过, so those clips would name the wrong actions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"sounds": {}, "music": {}}

    for name in SOUNDS:
        source = audio / "Sounds" / f"{name}.wav"
        if not source.exists():
            continue
        target = out_dir / f"{name}.opus"
        run_ffmpeg(source, target, ["-c:a", "libopus", "-b:a", "64k", "-vbr", "on"])
        manifest["sounds"][name] = {
            "file": f"audio/{name}.opus",
            "bytes": target.stat().st_size,
        }

    if music:
        for source in sorted((audio / "Music").glob("*.ogg")):
            target = out_dir / f"{source.stem}.opus"
            run_ffmpeg(source, target,
                       ["-c:a", "libopus", "-b:a", "96k", "-vbr", "on"])
            manifest["music"][source.stem] = {
                "file": f"audio/{source.stem}.opus",
                "bytes": target.stat().st_size,
            }

    return manifest


def run_ffmpeg(source: Path, target: Path, codec: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(source), *codec, str(target)],
        check=True,
    )


SUBSET_BASE = (
    "0123456789"
    "%+-/:.,()[]<>·×—…“”‘’、。！？；："
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
)


def collect_ui_charset(sources: list[Path]) -> str:
    """Return the CJK codepoints used by the supplied files and directories."""
    found: dict[str, None] = {}
    for source in sources:
        if not source.exists():
            continue
        paths = [source] if source.is_file() else sorted(source.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix not in (
                ".ts", ".tsx", ".html", ".css", ".json", ".md"
            ):
                continue
            for char in path.read_text(encoding="utf-8", errors="replace"):
                code = ord(char)
                if 0x2E80 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF or \
                   0xFF00 <= code <= 0xFFEF or 0x3000 <= code <= 0x303F:
                    found[char] = None
    return "".join(found)


def subset_font(source: Path, font_number: int, out_dir: Path,
                charset: str) -> dict:
    """Subset a CJK font down to the glyphs the UI actually renders."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "ui.woff2"
    unique = "".join(dict.fromkeys(SUBSET_BASE + charset))
    subprocess.run(
        ["pyftsubset", str(source),
         f"--font-number={font_number}",
         f"--text={unique}",
         "--layout-features=",
         "--no-hinting",
         "--desubroutinize",
         "--flavor=woff2",
         f"--output-file={target}"],
        check=True,
    )
    return {
        "file": "fonts/ui.woff2",
        "family": "BloodflowUI",
        "glyphs": len(unique),
        "bytes": target.stat().st_size,
        "source": source.name,
    }


SYSTEM_CJK_FONTS = (
    ("/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc", 2),
    ("/usr/share/fonts/google-noto-cjk-fonts/NotoSansCJKsc-Regular.otf", 0),
)

MARKER_GOLD = (198, 158, 62, 255)
MARKER_INK = (26, 20, 10, 255)
MARKER_DEALER_BG = (168, 46, 42, 255)
MARKER_INK_LIGHT = (250, 242, 220, 255)


def pick_marker_font() -> tuple[str, int]:
    for path, number in SYSTEM_CJK_FONTS:
        if Path(path).exists():
            return path, number
    return "", 0


def build_markers(out_dir: Path, font_path: str, font_number: int) -> dict:
    """Render seat markers for the wind_indicator plate.

    OpenRiichi's East/South/West/North kanji describe seat winds, which Blood
    Flow does not use. The same plate instead carries the declared missing suit
    (万 / 条 / 筒) and the dealer mark (庄).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    glyphs = {
        "marker_man": ("万", MARKER_GOLD, MARKER_INK),
        "marker_sou": ("条", MARKER_GOLD, MARKER_INK),
        "marker_pin": ("筒", MARKER_GOLD, MARKER_INK),
        "marker_dealer": ("庄", MARKER_DEALER_BG, MARKER_INK_LIGHT),
    }
    size = 256
    font = (
        ImageFont.truetype(font_path, 168, index=font_number)
        if font_path
        else ImageFont.load_default(168)
    )

    manifest: dict = {}
    for key, (glyph, background, ink) in glyphs.items():
        image = Image.new("RGBA", (size, size), background)
        draw = ImageDraw.Draw(image)
        box = draw.textbbox((0, 0), glyph, font=font)
        draw.text(
            ((size - box[2] - box[0]) / 2, (size - box[3] - box[1]) / 2),
            glyph, font=font, fill=ink,
        )
        image.save(out_dir / f"{key}.webp", "WEBP", lossless=True, method=6)
        manifest[key] = {"file": f"textures/{key}.webp", "glyph": glyph}
    return manifest


NOTICE = (
    "Derived from OpenRiichi (https://github.com/FluffyStuff/OpenRiichi), "
    "Copyright (C) FluffyStuff, licensed GPL-3.0-or-later. "
    "Converted for the Blood Flow Mahjong web client (AGPL-3.0-only)."
)

MODELS = ("table_high", "table_low", "table_center", "field", "field_tile",
          "wind_indicator", "stick")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/tmp/openriichi_clone/bin/Data"),
                        help="OpenRiichi bin/Data directory")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "public" / "assets")
    parser.add_argument("--music", action="store_true", help="also convert the two music tracks")
    parser.add_argument(
        "--charset-source",
        action="append",
        default=[],
        type=Path,
        help="additional file or directory whose CJK text must be included in the font",
    )
    args = parser.parse_args(argv)

    data = args.source
    if not (data / "Models" / "tile_high.obj").exists():
        print(f"error: {data} does not look like OpenRiichi bin/Data", file=sys.stderr)
        return 2

    out = args.out
    manifest: dict = {
        "manifestVersion": 1,
        "notice": NOTICE,
        "models": {},
        "textures": {},
    }

    for quality in ("high", "low"):
        manifest["models"][f"tile_{quality}"] = build_tile_glb(
            data / "Models", out / "models", quality, NOTICE
        )
    for name in MODELS:
        manifest["models"][name] = build_model_glb(
            data / "Models", out / "models", name, NOTICE
        )

    manifest["atlas"] = build_tile_atlas(data / "Textures", out / "textures")
    manifest["textures"].update(convert_textures(data / "Textures", out / "textures"))

    font_path, font_number = pick_marker_font()
    manifest["textures"].update(build_markers(out / "textures", font_path, font_number))

    manifest["audio"] = convert_audio(data / "Audio", out / "audio", args.music)

    web_root = Path(__file__).resolve().parents[1]
    repo_root = web_root.parent
    charset = collect_ui_charset([
        repo_root / "GAME_RULES.md",
        web_root / "src",
        web_root / "index.html",
        *args.charset_source,
    ])
    font_source = Path(font_path) if font_path else data / "Fonts" / "NotoSansCJKjp-Regular.otf"
    manifest["font"] = subset_font(font_source, font_number, out / "fonts", charset)

    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out / 'manifest.json'}")
    print(f"  models   {len(manifest['models'])}")
    print(f"  textures {len(manifest['textures'])} + atlas {ATLAS_W}x{ATLAS_H}")
    print(f"  sounds   {len(manifest['audio']['sounds'])}")
    print(f"  font     {manifest['font']['glyphs']} glyphs, "
          f"{manifest['font']['bytes'] / 1024:.1f}KB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
