"""Import the MCU card dataset into ``data/mcu_cards.json``.

Input: the folder created by ``mikroe_mcu_cards_ocr/build.py`` (data taken
from mikroe.com)::

    index.json                 layout groups, rules
    cards/<slug>_ocr.json      one file per card (specs, printed text,
                               look-alike cards, supported dev boards)

Output: one compact file that the catalog loads (``Catalog.from_csv``), so
the recogniser doesn't need the download folder at run time::

    {"groups": {group_id: {title, aliases, generation, family, number,
                           port_width, pins, verified}},
     "cards":  [{key, name, pid, group, mcu, mcus, empty, pins, flash_kb,
                 ram_bytes, vendor, architecture, generation,
                 supported_boards: [{name, pid}]}]}

``pins`` are the port pin labels next to mikroBUS 1 in the group's shop
photo (e.g. ATmega2560 cards: PD2 PD3 PD0 PD1, ATmega6450: PE0 PE1 PE4 PE5).
Uncertain reads (``RD15?``) are dropped. These tell apart groups that print
the same title.

Every group also gets ``text``: everything the dataset says is printed on
the card, used to match all of the card's text (``vision/card_text.py``)::

    {"pin_map":  {"MIKROBUS 1": {"AN": "PB0", ...}, ...},   # 8th gen, where read
     "port_pins": [...],        # every port pin label on the card
     "strings":   [...],        # other printed text (CAN, VLCD JMP, HW REV 1.01 ...)
     "header_pins": [...],      # older cards: port labels on the 4 pin-header rows
     "pcb": "black" | "green", "frame": "yellow" | None}

And every card gets ``markings``: the text on top of the chip as read from
the shop photo, joined into the strings an OCR engine will see
('ARM / STM32F405 / RGT6 / GH22B' -> 'STM32F405RGT6'; 'NXP logo / MK64 /
FN1M0V12' -> 'MK64FN1M0V12', which is NOT the ordering part number
MK64FN1M0VDC12).

    python scripts/import_mcu_cards.py ~/Downloads/mikroe_mcu_cards_ocr
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

_PIN = re.compile(r"^(?:[A-Z]{1,3}\d{1,3}(?:\.\d)?|ADC_A\d|PT[A-E]\d{1,2}|GP\d{3})$")


def parse_title(title: str):
    """Split a card title into parts.

    'MCU CARD 29 for STM32' -> ('card', 29, 'STM32')
    'SiBRAIN for PIC24/dsPIC33' -> ('sibrain', None, 'PIC24/dsPIC33')
    Older titles -> None.
    """
    m = re.match(r"^(MCU CARD|SiBRAIN)\s+(?:(\d+)\s+)?for\s+(\S+)$", title.strip())
    if not m:
        return None
    return ("card" if m.group(1) == "MCU CARD" else "sibrain",
            int(m.group(2)) if m.group(2) else None, m.group(3))


def fingerprint_pins(fp: str | None) -> list[str]:
    if not fp:
        return []
    m = re.search(r"MB1 (?:col|bottom)\s+([^;]+)", fp)
    if not m:
        return []
    out = []
    for t in re.split(r"[,\s]+", m.group(1)):
        t = t.strip().upper()
        if t and "?" not in t and _PIN.match(t) and t not in out:
            out.append(t)
    return out


_LOGO = re.compile(r"\b(?:ARM|NXP|TI|ST|MICROCHIP|ATMEL|logo|e3)\b", re.I)
_NOT_PIN = {"VCC", "GND", "NC", "DP", "DN", "DM", "MCLR", "RST", "VCAP", "VREF+", "VREF-",
            "MISO", "MOSI", "SCK", "VBAT"}


def marking_patterns(observed: str | None, quality: str | None, part: str | None) -> list[str]:
    """Turn a chip marking from a shop photo into the strings to look for on the chip.

    Lines are separated by ' / '. The first line after the vendor logo is
    always part of the marking ('M20A7V' is a Kinetis short code that has
    almost nothing in common with MK20DX128VFM5). The next lines are joined
    on as long as they share at least 3 characters with the part number
    (date and lot codes stop it). Unreadable ('...') lines stop it too.
    WRONG_CHIP or blank photos give nothing.
    """
    if not observed or quality in ("WRONG_CHIP", "none"):
        return []
    pc = re.sub(r"[^A-Z0-9]", "", (part or "").upper())

    def lcs(a: str, b: str) -> int:
        best = 0
        for i in range(len(a)):
            for j in range(i + best + 1, len(a) + 1):
                if a[i:j] in b:
                    best = j - i
                else:
                    break
        return best

    if re.match(r"^\s*(none|\(blank)", observed, re.I):
        return []
    slashed = " / " in observed
    lines = []
    for ln in (observed.split(" / ") if slashed else observed.split()):
        ln = re.sub(r"\(.*?\)", "", ln)
        ln = _LOGO.sub("", ln).strip()
        if not ln:
            continue
        if "..." in ln or "?" in ln:
            lines.append(None)  # unreadable: stops the joining
            continue
        lines.append(re.sub(r"[^A-Z0-9]", "", ln.upper()))
    if not slashed:  # single-line older reading: start at the part number
        i = next((i for i, ln in enumerate(lines) if ln and pc and lcs(ln, pc) >= 3), None)
        lines = lines[i:] if i is not None else []

    def rest(cur: str) -> str:
        """Return what's left of the part number after the longest prefix ``cur`` covers."""
        k = 0
        while k < min(len(cur), len(pc)) and cur[k] == pc[k]:
            k += 1
        if k < 3:  # '32F215...' / 'M20A7V': line up on the end of cur
            k = max((pc.find(cur[-j:]) + j for j in range(min(len(cur), 6), 2, -1)
                     if cur[-j:] in pc), default=len(pc))
        return pc[k:]

    out, cur = [], ""
    for ln in lines:
        if ln is None:
            break
        if not cur:
            cur = ln
            continue
        r = rest(cur)
        if pc and len(r) >= 2 and lcs(ln, r) >= 2:
            cur += ln
        else:
            break
    if len(cur) >= 5:
        out.append(cur)
    return out


def text_profile(recs: list[dict], fp_pins: list[str]) -> dict:
    """Collect everything printed on the cards of one layout group.

    Combines all the group's card files, since some contain more of the
    shop photo reading than others.
    """
    pin_map, port_pins, strings, header_pins = {}, [], [], []
    pcb = frame = None

    def add(lst, x):
        if x and x not in lst:
            lst.append(x)

    for d in recs:
        top = d["ocr_strings"]["top_side"]
        for mb, sig in (top.get("mikrobus_pin_map") or {}).items():
            if mb not in pin_map:
                pin_map[mb] = {k: v.upper() for k, v in sig.items()
                               if v and "?" not in v and _PIN.match(v.upper())}
        for p in top.get("port_pin_labels") or []:
            if "?" not in p and _PIN.match(p.upper()):
                add(port_pins, p.upper())
        for st in (top.get("other_strings") or []) + (top.get("observed_strings") or []):
            st = st.strip()
            if len(st) >= 3 and "logo" not in st.lower() and st.upper() != top.get("title", "").upper():
                add(strings, st)
        for row in (top.get("header_pin_labels") or {}).values():
            for p in row.split():
                p = p.upper()
                if p not in _NOT_PIN and "?" not in p and _PIN.match(p):
                    add(header_pins, p)
        vf = d.get("visual_features") or {}
        col = (vf.get("pcb_color") or "").lower()
        pcb = pcb or ("black" if "black" in col else "green" if "green" in col else None)
        frame = frame or ("yellow" if "yellow" in (vf.get("edge") or "").lower() else None)
    for sig in pin_map.values():
        for p in sig.values():
            add(port_pins, p)
    for p in fp_pins:
        add(port_pins, p)
    return {"pin_map": pin_map, "port_pins": port_pins, "strings": strings,
            "header_pins": header_pins, "pcb": pcb, "frame": frame}


def clean_part(p: str) -> str | None:
    """Clean up an entry from the compatible-MCU list.

    Drops '-PT' package suffixes and family wildcards ('PIC24FJXXGA004'
    isn't a part number).
    """
    p = p.strip().upper().split("-")[0]
    if not p or "XX" in p or not re.search(r"\d", p):
        return None
    return p


def build(src: Path) -> dict:
    index = json.loads((src / "index.json").read_text(encoding="utf-8"))
    groups, cards = {}, []
    for gid, g in index["groups"].items():
        aliases: list[str] = []
        recs = []
        for c in g["cards"]:
            d = json.loads((src / c["file"]).read_text(encoding="utf-8"))
            recs.append(d)
            for a in d["ocr_strings"]["top_side"].get("title_aliases") or []:
                if a not in aliases:
                    aliases.append(a)
        top = recs[0]["ocr_strings"]["top_side"]
        title = g["title"]
        legacy = parse_title(title) is None
        if legacy:   # "LV18F v6 / mikroElektronika": the logo text is the first part
            title = title.split(" / ")[0].strip()
        parsed = parse_title(title)
        groups[gid] = {
            "title": title,
            "aliases": aliases,
            "generation": g["generation"],
            "family": parsed[2] if parsed else None,
            "number": parsed[1] if parsed else None,
            "port_width": top.get("port_width_box"),
            "pins": fingerprint_pins(g.get("fingerprint")),
            "verified": bool(g.get("verified_from_photo")),
            "text": text_profile(recs, fingerprint_pins(g.get("fingerprint"))),
        }
        for d in recs:
            s = d["specs"]
            mcu = (s.get("mcu") or "").upper() or None
            compat = [p for p in (clean_part(x) for x in d.get("compatible_mcus_for_empty_card") or []) if p]
            mcus = [mcu] if mcu else []
            mcus += [p for p in compat if p not in mcus]
            cards.append({
                "key": d["card_id"], "name": d["product_name"], "pid": d["pid"], "group": gid,
                "mcu": mcu, "mcus": mcus, "empty": mcu is None,
                "pins": s.get("pin_count"), "flash_kb": s.get("flash_kb"),
                "ram_bytes": s.get("ram_bytes"), "vendor": s.get("silicon_vendor"),
                "architecture": s.get("architecture"), "generation": s.get("type"),
                "supported_boards": d.get("supported_dev_boards") or [],
                "marking": d["ocr_strings"]["top_side"].get("chip_marking_observed"),
                "markings": marking_patterns(
                    d["ocr_strings"]["top_side"].get("chip_marking_observed"),
                    d["ocr_strings"]["top_side"].get("chip_marking_quality"), mcu),
            })
    return {"source": index.get("source"), "generated": index.get("generated"),
            "imported_from": "mikroe_mcu_cards_ocr (build.py)",
            "card_count": len(cards), "group_count": len(groups),
            "groups": groups, "cards": cards}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path, help="mikroe_mcu_cards_ocr folder (index.json + cards/)")
    ap.add_argument("--out", type=Path, default=DATA / "mcu_cards.json")
    a = ap.parse_args(argv)
    data = build(a.src)
    a.out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_mcu = len({m for c in data["cards"] for m in c["mcus"]})
    print(f"{data['card_count']} cards, {data['group_count']} groups, {n_mcu} MCU parts "
          f"-> {a.out}")


if __name__ == "__main__":
    main()
