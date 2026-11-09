"""Exact accelerations for the fuzzy text matcher (efficiency study, hypothesis H-006).

The profile of the reference benchmark (docs/reports/efficiency.md) shows
``vision.textmatch.ocr_distance`` - a pure-Python weighted Levenshtein - called
23 million times for 65 photos. It is a pure function of two strings, and the
same (catalogue word, OCR word) pairs come back for every rotation, variant,
pass and photo. Memoising it returns bit-identical results, so every decision
must stay identical: that is the gate for this option.

Installed by ``PhoenixPipeline`` when the registry stage ``text_match`` is
``exact_cached``. It replaces module attributes of ``vision.textmatch`` at run
time (callers that imported ``ocr_ratio`` by value still call the patched
``ocr_distance`` through the module global), and ``uninstall`` restores them.
"""

from __future__ import annotations

import functools

import vision.textmatch as TM

_ORIG = {}
CACHE_SIZE = 1 << 20  # ~1M pairs, a few hundred MB at most; cleared per process


def install(mode: str = "exact_cached") -> None:
    if mode == "exact" or _ORIG:
        return
    for name in ("ocr_distance", "normalize", "compact"):
        _ORIG[name] = getattr(TM, name)
        setattr(TM, name, functools.lru_cache(maxsize=CACHE_SIZE)(_ORIG[name]))


def uninstall() -> None:
    for name, fn in _ORIG.items():
        setattr(TM, name, fn)
    _ORIG.clear()


def stats() -> dict:
    out = {}
    for name in ("ocr_distance", "normalize", "compact"):
        fn = getattr(TM, name)
        if hasattr(fn, "cache_info"):
            ci = fn.cache_info()
            out[name] = {"hits": ci.hits, "misses": ci.misses,
                         "hit_rate": round(ci.hits / max(1, ci.hits + ci.misses), 3)}
    return out
