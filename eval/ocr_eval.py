"""Test how well OCR identifies boards, MCUs and cards. Writes eval/ocr_eval.md.

Folder layout:
    photos/
      labels.csv     filename,kind,label[,distance_cm]
                     kind = board | mcu | mcu_card
                     label = board key or name / MCU part / card key
      *.jpg

Usage:
    python eval/ocr_eval.py photos/ [--backend tesseract] [--md eval/ocr_eval.md]
    python eval/ocr_eval.py photos/ --configs baseline,full          # compare settings
    python eval/ocr_eval.py photos/ --degrade none,defocus4,motion9,dark,noise12,jpeg25,half

For each kind it reports: correct / wrong / gave up, accuracy on all
photos, accuracy when it did answer (precision), and how often the right
label was in the top 3. It also reports the character / word error rate
(CER / WER) of the raw text behind each answer, compares settings
(``--configs``), and tests robustness by applying fake damage (``--degrade``)
to the same real photos. Plus: accuracy per distance, and the typical
letter height of correctly read part numbers (so you know how close to
hold the camera).

Configs:
    baseline        no crop re-read step
    full            the default pipeline = baseline + crop re-read (vision/refine.py)
    preproc         baseline + whole-photo denoise / brighten / sharpen (optional)
    preproc+refine  both
    sr-<name>       full, with crops enlarged by a super-resolution model (espcn, fsrcnn, edsr)
    autoquad        full + automatic board perspective correction
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.catalog import Catalog  # noqa: E402
from vision.loader import Photo, iter_photos, split_hash  # noqa: E402
from vision.pipeline import Pipeline  # noqa: E402
from vision.preprocess import PreprocessConfig  # noqa: E402
from vision.refine import RefineConfig  # noqa: E402
from vision.textmatch import compact, levenshtein, normalize, words  # noqa: E402


# ---- Metrics ----
def cer(ref: str, hyp: str, prefix: bool = False) -> float:
    """Character error rate.

    ``prefix``: the read text may include an ordering suffix
    (STM32F407ZGT6 for STM32F407ZG), so compare against its best prefix.
    """
    if not ref:
        return 0.0
    if not hyp:
        return 1.0
    if prefix and len(hyp) > len(ref):
        return min(levenshtein(ref, hyp[:k]) for k in range(max(1, len(ref) - 2),
                                                            min(len(hyp), len(ref) + 2) + 1)) / len(ref)
    return levenshtein(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    if not r:
        return 0.0
    # edit distance on words
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def best_read(kind: str, rec) -> str:
    """Return the raw OCR text behind the answer (or behind the best guess)."""
    c = rec.decision or (rec.candidates[0] if rec.candidates else None)
    if kind == "mcu":
        if c is not None and c.detail.get("read"):
            return c.detail["read"]
        return rec.raw_reads[0] if rec.raw_reads else ""
    if c is not None and c.detail.get("line"):
        return c.detail["line"]
    return rec.raw_reads[0] if rec.raw_reads else ""


def same(kind: str, pred, label: str) -> bool:
    if pred is None:
        return False
    if kind == "mcu":
        return compact(pred.key) == compact(label)
    return normalize(label) in {normalize(pred.key), normalize(pred.name)}


# ---- Fake damage: blur, noise, low light, etc., to see how much recognition suffers ----
def degrade(img: np.ndarray, name: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if name in ("", "none"):
        return img
    if name.startswith("blur"):
        return cv2.GaussianBlur(img, (0, 0), float(name[4:] or 2))
    if name.startswith("defocus"):
        r = int(name[7:] or 4)
        k = np.zeros((2 * r + 1,) * 2, np.float32)
        cv2.circle(k, (r, r), r, 1, -1)
        return cv2.filter2D(img, -1, k / k.sum())
    if name.startswith("motion"):
        n = int(name[6:] or 9)
        k = np.zeros((n, n), np.float32)
        k[n // 2, :] = 1
        return cv2.filter2D(img, -1, k / k.sum())
    if name.startswith("noise"):
        s = float(name[5:] or 10)
        return np.clip(img + rng.normal(0, s, img.shape), 0, 255).astype(np.uint8)
    if name == "dark":
        d = img.astype(np.float32) * 0.25
        return np.clip(d + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    if name.startswith("jpeg"):
        q = int(name[4:] or 25)
        return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1],
                            cv2.IMREAD_COLOR)
    if name in ("half", "quarter"):
        f = 0.5 if name == "half" else 0.25
        return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    if name.startswith("persp"):
        a = float(name[5:] or 0.08)
        h, w = img.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.uniform(-a, a, (4, 2)).astype(np.float32) * [w, h]
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                                   borderMode=cv2.BORDER_REPLICATE)
    raise ValueError(f"unknown degradation {name!r}")


# ---- Settings to compare ----
GLOBAL_ON = dict(denoise="auto", lift_dark=True, sharpen="auto")


def make_pipeline(name: str, catalog, backend: str) -> Pipeline:
    if name == "baseline":
        return Pipeline(catalog, backend, refine="off")
    if name == "full":
        return Pipeline(catalog, backend)
    if name == "preproc":
        return Pipeline(catalog, backend, PreprocessConfig(**GLOBAL_ON), refine="off")
    if name == "preproc+refine":
        return Pipeline(catalog, backend, PreprocessConfig(**GLOBAL_ON))
    if name.startswith("sr-"):
        return Pipeline(catalog, backend, refine_cfg=RefineConfig(upscaler=name[3:]))
    if name == "autoquad":
        return Pipeline(catalog, backend, PreprocessConfig(auto_perspective=True))
    raise ValueError(f"unknown config {name!r}")


def run(pipe: Pipeline, photos: list[Photo], deg: str, catalog: Catalog):
    names = {b.key: b.name for b in catalog.boards}
    stats = defaultdict(lambda: {"n": 0, "ok": 0, "wrong": 0, "abst": 0, "top3": 0,
                                 "cer": [], "wer": []})
    by_dist = defaultdict(lambda: [0, 0])
    glyphs, rows, latencies = [], [], []
    for p in photos:
        kind, label = p.label["kind"].strip(), p.label["label"].strip()
        ph = p if deg in ("", "none") else Photo(**{**p.__dict__,
                                                    "image": degrade(p.image, deg)})
        t0 = time.perf_counter()
        res = pipe.run(ph)
        latencies.append((time.perf_counter() - t0) * 1000)
        rec = {"board": res.board, "mcu": res.mcu, "mcu_card": res.mcu_card}[kind]
        s = stats[kind]
        s["n"] += 1
        open_set = label.lower() in ("none", "unknown")
        if open_set:
            verdict = "ok" if rec.decision is None else "WRONG"
            s["ok" if verdict == "ok" else "wrong"] += 1
        elif rec.decision is None:
            s["abst"] += 1
            verdict = "abstain"
        elif same(kind, rec.decision, label):
            s["ok"] += 1
            verdict = "ok"
            if kind == "mcu" and rec.decision.detail.get("glyph_px"):
                glyphs.append(rec.decision.detail["glyph_px"])
        else:
            s["wrong"] += 1
            verdict = "WRONG"
        in_top3 = any(same(kind, c, label) for c in rec.candidates)
        s["top3"] += in_top3 or open_set
        read = best_read(kind, rec)
        if not open_set:
            if kind == "mcu":
                s["cer"].append(cer(compact(label), compact(read), prefix=True))
            else:
                ref = names.get(label, label)
                s["cer"].append(cer(compact(ref), compact(read)))
                s["wer"].append(wer(ref, read))
        dist = (p.label.get("distance_cm") or "").strip()
        if dist:
            by_dist[(kind, dist)][0] += 1
            by_dist[(kind, dist)][1] += verdict == "ok"
        rows.append((p.name, kind, label, rec.decision.name if rec.decision else "—",
                     verdict, "yes" if in_top3 else "no", read or "—", rec.reason))
    return stats, by_dist, glyphs, rows, latencies


def _mean(xs):
    return f"{statistics.mean(xs):.1%}" if xs else "—"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--configs", default="full",
                    help="comma list: baseline, full, preproc, preproc+refine, sr-espcn, autoquad")
    ap.add_argument("--degrade", default="none",
                    help="comma list: none, blurN, defocusN, motionN, noiseN, dark, jpegQ, "
                         "half, quarter, perspA")
    ap.add_argument("--md", type=Path, default=Path(__file__).with_name("ocr_eval.md"))
    a = ap.parse_args(argv)

    photos = [p for p in iter_photos(a.folder) if p.label and
              p.label.get("kind", "").strip() in ("board", "mcu", "mcu_card")]
    if not photos:
        print("no labelled photos (need labels.csv with filename,kind,label)")
        return 1
    catalog = Catalog.load(a.catalog)
    configs = [c.strip() for c in a.configs.split(",") if c.strip()]
    degs = [d.strip() for d in a.degrade.split(",") if d.strip()]

    L = [f"# OCR evaluation — {date.today().isoformat()}", "",
         f"Photos: {len(photos)} · split hash: `{split_hash(photos)}`"]
    summary = []  # (config, damage, kind, stats, time)
    first = None
    for cfg in configs:
        pipe = make_pipeline(cfg, catalog, a.backend)
        for deg in degs:
            print(f"[eval] config={cfg} degrade={deg} ...", flush=True)
            out = run(pipe, photos, deg, catalog)
            if first is None:
                first = (cfg, deg, pipe.backend.name, out)
            stats, _, _, _, lat = out
            for kind, s in stats.items():
                summary.append((cfg, deg, kind, s, statistics.median(lat)))

    cfg0, deg0, backend, (stats, by_dist, glyphs, rows, latencies) = first
    L[-1] += f" · backend: `{backend}`"
    L += ["", f"## Main result (config `{cfg0}`, degradation `{deg0}`)", "",
          "| kind | n | correct | wrong | abstained | accuracy | precision (when answering) "
          "| label in top-3 | CER | WER |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for kind, s in sorted(stats.items()):
        answered = s["ok"] + s["wrong"]
        L.append(f"| {kind} | {s['n']} | {s['ok']} | {s['wrong']} | {s['abst']} | "
                 f"{s['ok'] / s['n']:.0%} | {(s['ok'] / answered) if answered else 0:.0%} | "
                 f"{s['top3'] / s['n']:.0%} | {_mean(s['cer'])} | {_mean(s['wer'])} |")
    L.append(f"\nMedian latency {statistics.median(latencies):.0f} ms "
             f"(max {max(latencies):.0f} ms).")
    if len(summary) > len(stats):
        L += ["", "## Ablation / robustness", "",
              "CER = character error rate of the raw read behind the answer "
              "(lower is better).", "",
              "| config | degradation | kind | n | accuracy | wrong | abstained | CER | median ms |",
              "|---|---|---|---|---|---|---|---|---|"]
        for cfg, deg, kind, s, lat in summary:
            L.append(f"| {cfg} | {deg} | {kind} | {s['n']} | {s['ok'] / s['n']:.0%} | "
                     f"{s['wrong']} | {s['abst']} | {_mean(s['cer'])} | {lat:.0f} |")
    if by_dist:
        L += ["", "## By photo distance", "", "| kind | distance (cm) | n | correct |",
              "|---|---|---|---|"]
        for (kind, d), (n, ok) in sorted(by_dist.items(),
                                        key=lambda kv: (kv[0][0], float(kv[0][1]))):
            L.append(f"| {kind} | {d} | {n} | {ok} ({ok / n:.0%}) |")
    if glyphs:
        L += ["", f"Correct MCU reads had glyph height median **{statistics.median(glyphs):.0f} px**"
              f" (min {min(glyphs):.0f} px) in the preprocessed image."]
    L += ["", "## Per photo", "",
          "| photo | kind | label | predicted | result | label in top-3 | raw read | reason |",
          "|---|---|---|---|---|---|---|---|"]
    L += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    a.md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:14]))
    print(f"\nwritten: {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
