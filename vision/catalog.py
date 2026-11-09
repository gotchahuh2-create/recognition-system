"""The lists of known boards, MCUs, cards and Clicks that the recognisers match against.

There are two possible sources. Both load into the same structure:

* The CSV files in ``data/`` (boards.csv, mcus.csv, mcu_cards.csv,
  clicks.csv, sockets.csv) plus ``data/mcu_cards.json`` (207 MCU cards in
  62 layout groups, built from the mikroe.com website data with
  ``scripts/import_mcu_cards.py``). These are a temporary stand-in for
  the real product database and haven't been checked against it yet.
* ``products.db``, a SQLite product database with this table:
      products(pid PK, name, type [click|board|mcu_card|mcu], slug, ..., family, status)
  For MCU cards, the MCU comes from the ``compat`` table if there is one,
  otherwise it's taken from the card's name.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .textmatch import compact

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Board:
    key: str
    name: str
    aliases: list[str] = field(default_factory=list)
    pid: str | None = None
    family: str | None = None


@dataclass
class Mcu:
    part: str                      # base part number, e.g. STM32F407ZG
    vendor: str | None = None
    family: str | None = None
    pid: str | None = None

    @property
    def compact(self) -> str:
        return compact(self.part)


@dataclass
class McuCard:
    key: str
    name: str
    mcu: str | None = None
    pid: str | None = None
    group: str | None = None  # layout group (from data/mcu_cards.json)
    # MCUs this card can carry: the one soldered on it, or for an empty card
    # the parts listed as compatible (as catalog part numbers)
    mcus: list[str] = field(default_factory=list)
    empty: bool = False
    # keys of the dev boards that support this card (empty if unknown)
    boards: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)   # specs (pins, flash, vendor ...)
    # the text on top of the chip as seen in the shop photo, joined the way OCR
    # reads it ("MK64FN1M0V12" for MK64FN1M0VDC12, "32F215RET6" for STM32F215RE)
    markings: list[str] = field(default_factory=list)


@dataclass
class CardGroup:
    """Cards that share one printed layout.

    For example "MCU CARD 29 for STM32" is printed on 10 different cards.
    Reading the title tells us the GROUP. Only the chip marking (or a PID
    label) can tell the cards in a group apart.
    """
    key: str
    title: str
    aliases: list[str] = field(default_factory=list)
    generation: str | None = None
    family: str | None = None  # from an 8th-gen title, e.g. STM32, PIC24/dsPIC33
    number: int | None = None  # "MCU CARD 29 ..." -> 29, None if there is no number
    port_width: str | None = None  # "16-BIT PORT" or "8-BIT PORT"
    pins: list[str] = field(default_factory=list)  # port pin labels next to mikroBUS 1
    verified: bool = False  # True if the title was read from the shop photo, not guessed
    cards: list[str] = field(default_factory=list)  # card keys
    # everything printed on the card (see scripts/import_mcu_cards.py): pin_map,
    # port_pins, strings, header_pins, PCB colour, frame colour
    text: dict = field(default_factory=dict)


@dataclass
class Click:
    pid: str                       # MIKROE-1234
    name: str                      # "Thermo 2 Click"
    size: str | None = None  # S, M or L (mikroBUS Click sizes)
    group: str | None = None  # boards that look almost the same
    aliases: list[str] = field(default_factory=list)


@dataclass
class Socket:
    board: str                     # board key
    slot: str                      # "mikroBUS 1"
    u: float  # socket centre, 0..1 across the board
    v: float


@dataclass
class Catalog:
    boards: list[Board]
    mcus: list[Mcu]
    mcu_cards: list[McuCard]
    clicks: list[Click] = field(default_factory=list)
    sockets: list[Socket] = field(default_factory=list)
    card_groups: dict[str, CardGroup] = field(default_factory=dict)

    # ---- Loading from CSV ----
    @classmethod
    def from_csv(cls, data_dir: str | Path = DATA_DIR) -> "Catalog":
        d = Path(data_dir)

        def rows(name):
            p = d / name
            if not p.is_file():
                return []
            with open(p, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))

        boards = [Board(r["key"], r["name"],
                        [a for a in (r.get("aliases") or "").split("|") if a],
                        r.get("pid") or None, r.get("family") or None)
                  for r in rows("boards.csv")]
        mcus = [Mcu(r["part"].strip().upper(), r.get("vendor") or None,
                    r.get("family") or None, r.get("pid") or None)
                for r in rows("mcus.csv")]
        cards = [McuCard(r["key"], r["name"], (r.get("mcu") or "").upper() or None,
                         r.get("pid") or None)
                 for r in rows("mcu_cards.csv")]
        clicks = [Click(r["pid"].strip().upper(), r["name"].strip(),
                        (r.get("size") or "").strip().upper() or None,
                        (r.get("group") or "").strip() or None,
                        [a for a in (r.get("aliases") or "").split("|") if a])
                  for r in rows("clicks.csv") if (r.get("pid") or "").strip()]
        sockets = []
        for r in rows("sockets.csv"):
            try:
                sockets.append(Socket(r["board"], r["slot"], float(r["u"]), float(r["v"])))
            except (KeyError, ValueError):
                continue
        cat = cls(boards, mcus, cards, clicks, sockets)
        if (d / "mcu_cards.json").is_file():
            cat.add_card_dataset(d / "mcu_cards.json")
        for c in cat.mcu_cards:  # CSV cards have exactly one MCU
            if c.mcu and not c.mcus:
                c.mcu = cat.canonical_part(c.mcu, add=True)
                c.mcus = [c.mcu]
        return cat

    # ---- Loading the card dataset ----
    def add_card_dataset(self, path: str | Path) -> None:
        """Load ``data/mcu_cards.json`` (made by scripts/import_mcu_cards.py).

        This adds the card layout groups, the cards, their MCUs (added to
        ``mcus``) and the dev boards they support (matched to board keys by name).
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for gk, g in data.get("groups", {}).items():
            self.card_groups[gk] = CardGroup(
                gk, g["title"], list(g.get("aliases") or []), g.get("generation"),
                g.get("family"), g.get("number"), g.get("port_width"),
                [p.upper() for p in g.get("pins") or []], bool(g.get("verified")),
                text=dict(g.get("text") or {}))
        known = {compact(c.name) for c in self.mcu_cards}
        # Shorter part numbers first, so "STM32F407VGT6" (from an old card name)
        # merges into "STM32F407VG" instead of becoming a second entry.
        parts = sorted({p for c in data.get("cards", []) for p in c.get("mcus") or []},
                       key=len)
        vendor = {}
        for c in data.get("cards", []):
            for p in c.get("mcus") or []:
                vendor.setdefault(p, c.get("vendor"))
        canon = {p: self.canonical_part(p, add=True, vendor=vendor.get(p)) for p in parts}
        for c in data.get("cards", []):
            if compact(c["name"]) in known:
                continue
            mcus = list(dict.fromkeys(canon[p] for p in c.get("mcus") or []))
            card = McuCard(c["key"], c["name"], canon.get(c.get("mcu")) if c.get("mcu") else None,
                           c.get("pid"), c.get("group"), mcus, bool(c.get("empty")),
                           self._board_keys(b["name"] for b in c.get("supported_boards") or []),
                           {k: c.get(k) for k in ("pins", "flash_kb", "ram_bytes", "vendor",
                                                  "architecture", "generation")},
                           [compact(m) for m in c.get("markings") or []])
            self.mcu_cards.append(card)
            if card.group in self.card_groups:
                self.card_groups[card.group].cards.append(card.key)

    def canonical_part(self, part: str, add: bool = False, vendor: str | None = None) -> str:
        """Return the catalog part number for ``part``.

        An ordering suffix of up to 3 characters is still the same chip
        ("STM32F407VGT6" = "STM32F407VG", "TM4C123GH6PMI" = "TM4C123GH6PM").
        With ``add=True``, unknown parts are added to ``mcus``.
        """
        c = compact(part)
        for m in self.mcus:
            mc = m.compact
            if mc == c:
                return m.part
            short, long_ = (mc, c) if len(mc) < len(c) else (c, mc)
            if len(short) >= 8 and long_.startswith(short) and len(long_) - len(short) <= 3:
                return m.part
        if add:
            self.mcus.append(Mcu(part.strip().upper(), vendor, None, None))
            return self.mcus[-1].part
        return part.strip().upper()

    def _board_keys(self, names) -> list[str]:
        """Turn board names from the shop into catalog board keys.

        For example "UNI-DS v8 over USB-C" or "Fusion for PIC v8 with PIC18F97J94".
        """
        idx = {}
        for b in self.boards:
            for n in [b.name] + b.aliases:
                idx.setdefault(compact(n), b.key)
        out = []
        for n in names:
            base = re.sub(r"\s+over\s+USB-C$", "", n.strip(), flags=re.I)
            base = re.sub(r"\s+with\s+\S+$", "", base, flags=re.I)
            k = idx.get(compact(n)) or idx.get(compact(base))
            if k and k not in out:
                out.append(k)
        return out

    def board_has_card_data(self, board_key: str) -> bool:
        """True if any card lists this board as supported.

        Only then can the supported-boards list be used to rule a card out.
        """
        return any(board_key in c.boards for c in self.mcu_cards)

    def card_by_key(self, key: str) -> McuCard | None:
        return next((c for c in self.mcu_cards if c.key == key), None)

    # ---- Loading from SQLite ----
    @classmethod
    def from_sqlite(cls, db_path: str | Path) -> "Catalog":
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        try:
            prods = con.execute(
                "SELECT pid, name, type, slug, family FROM products "
                "WHERE type IN ('board','mcu','mcu_card','click') "
                "AND (status IS NULL OR status != 'discontinued')").fetchall()
            compat: dict[str, str] = {}
            try:
                for r in con.execute("SELECT board_pid, mcu_or_card_pid FROM compat"):
                    compat.setdefault(r[0], r[1])
            except sqlite3.OperationalError:
                pass
        finally:
            con.close()
        boards, mcus, cards, clicks = [], [], [], []
        for p in prods:
            if p["type"] == "click":
                clicks.append(Click(p["pid"], p["name"]))
            elif p["type"] == "board":
                boards.append(Board(p["slug"] or p["pid"], p["name"], [], p["pid"], p["family"]))
            elif p["type"] == "mcu":
                mcus.append(Mcu(p["name"].upper(), None, p["family"], p["pid"]))
            else:
                cards.append(McuCard(p["slug"] or p["pid"], p["name"],
                                     _mcu_from_card_name(p["name"]), p["pid"]))
        return cls(boards, mcus, cards, clicks)

    @classmethod
    def load(cls, source: str | Path | None = None) -> "Catalog":
        if source is None:
            return cls.from_csv()
        p = Path(source)
        return cls.from_sqlite(p) if p.suffix in {".db", ".sqlite"} else cls.from_csv(p)

    def click_names(self) -> dict[str, str]:
        return {c.pid: c.name for c in self.clicks}

    def click_by_pid(self, pid: str) -> Click | None:
        return next((c for c in self.clicks if c.pid == pid), None)

    def sockets_for(self, board_key: str) -> list[Socket]:
        return [s for s in self.sockets if s.board == board_key]

    def mcu_by_part(self, part: str) -> Mcu | None:
        c = compact(part)
        return next((m for m in self.mcus if m.compact == c), None)


_PART_RE = re.compile(r"\b((?:DS)?PIC\d\w+|STM32\w+|AT(?:MEGA|SAM|TINY|XMEGA)\w+|"
                      r"MK\d\w+|TM4C\w+|LPC\d\w+|GD32\w+|EFM32\w+|NRF\d\w+)\b", re.I)


def _mcu_from_card_name(name: str) -> str | None:
    m = _PART_RE.search(name)
    return m.group(1).upper() if m else None
