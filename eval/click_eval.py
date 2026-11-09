"""Test Click recognition, tune its thresholds, and review the failures.

    python eval/click_eval.py photos/clicks/ --gallery gallery/ \
        --model models/dinov2s_reg_224.onnx --exp E1 [--tta --ocr] \
        [--add-train] [--write-thresholds]

Photos: Click photos, labelled either by file name (``MIKROE-1234_phoneA_lamp_
20deg_01.jpg``) or by ``labels.csv`` rows ``filename,kind=click,label=PID
[,box=x,y,w,h][,session=...]``. Without a box, the whole photo is the crop.

Photos are split by the session they were taken in (the ``session`` column,
or device+lighting from the file name), using a hash so the split never
changes: train (can be added to the gallery with --add-train), val (used to
tune thresholds) and test (only for reporting results; never used for
tuning).

Each run reports: top-1, top-3, average recall per Click, accuracy on
look-alike pairs, how well unknown Clicks are rejected (10 % of PIDs are
removed from the gallery), the share of accept / ask / unknown, the
precision of automatic accepts, and the 95th percentile time per crop. One
row is added to ``eval/results.csv`` (experiments E1-E7), and an overview
image of the 20 worst failures (query | wrong #1 | correct reference) is
saved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.catalog import Catalog  # noqa: E402
from vision.clicks import ClickConfig, ClickRecognizer, Gallery, get_embedder  # noqa: E402
from vision.clicks.preprocess import crop_box  # noqa: E402
from vision.clicks.rerank import ocr_click, series_base  # noqa: E402
from vision.loader import iter_photos  # noqa: E402

RESULTS_COLS = ["date", "exp", "model_version", "preprocess_version", "gallery", "rerank",
                "split", "n", "pids", "top1", "top3", "macro_recall", "hard_pair_acc",
                "hard_n", "open_set_unknown", "open_set_false_accept", "accept", "ask",
                "unknown", "accept_precision", "t_accept", "t_unknown", "margin",
                "p95_ms", "split_hash"]


# ---- Data ----
def click_photos(folder):
    out = []
    for p in iter_photos(folder):
        lab = p.label or {}
        kind = (lab.get("kind") or "").strip()
        if lab and kind and kind != "click":
            continue
        pid = (lab.get("label") or "").strip().upper() if kind == "click" else p.pid
        if not pid:
            continue
        box = None
        if (lab.get("box") or "").strip():
            box = [float(v) for v in lab["box"].split(",")][:4]
        session = (lab.get("session") or "").strip() or \
            "_".join(x for x in (p.device, p.lighting) if x) or (p.group_id or "")
        out.append({"photo": p, "pid": pid, "box": box, "session": session or p.sha1})
    return out


def split_of(session: str, val: float, test: float) -> str:
    h = int(hashlib.sha1(session.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


# ---- Threshold tuning ----
def outcomes(rec: ClickRecognizer, items, cfg: ClickConfig):
    out = []
    for it in items:
        r = rec.recognise(it["image"], it["box"], q=it["q"], ocr_res=it["ocr"], cfg=cfg,
                          gallery=it.get("gallery"))
        out.append(r)
    return out


def calibrate(rec: ClickRecognizer, val, base: ClickConfig, target: float = 0.98) -> ClickConfig:
    """Tune the thresholds on the validation set.

    T_unknown: the 5th percentile of the correct #1 cosine score.
    M and T_accept: whatever gives the most automatic accepts while keeping
    precision at or above ``target``. The ranking doesn't depend on the
    thresholds, so each setting is just a cheap re-decision of the same
    ranked candidates.
    """
    res = outcomes(rec, val, base)
    correct_cos = [r.raw[0]["cos"] for r, it in zip(res, val)
                   if r.raw and r.raw[0]["pid"] == it["pid"]]
    t_unknown = round(float(np.percentile(correct_cos, 5)), 3) if correct_cos else base.t_unknown
    best = None
    for m in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1):
        for ta in np.arange(max(t_unknown, 0.2), 0.995, 0.01):
            cfg = replace(base, t_accept=round(float(ta), 3), t_unknown=t_unknown, margin=m)
            dec = [rec.redecide(r, cfg) for r in res]
            acc = [pid == it["pid"] for (oc, pid), it in zip(dec, val) if oc == "accept"]
            if acc and np.mean(acc) >= target:
                if best is None or len(acc) > best[0]:
                    best = (len(acc), cfg)
                break  # smallest T_accept that reaches the target for this M
    if best is None:
        print(f"[calibrate] no setting reaches {target:.0%} precision on val; keeping defaults")
        return replace(base, t_unknown=t_unknown)
    return best[1]


# ---- Metrics ----
def metrics(rec, items, results, hard_pids, ms):
    n = len(items)
    top1 = [bool(r.recognition.candidates) and r.recognition.candidates[0].pid == it["pid"]
            for r, it in zip(results, items)]
    top3 = [any(c.pid == it["pid"] for c in r.recognition.candidates[:3])
            for r, it in zip(results, items)]
    per = defaultdict(list)
    for ok, it in zip(top1, items):
        per[it["pid"]].append(ok)
    hard = [ok for ok, it in zip(top1, items) if it["pid"] in hard_pids]
    oc = [r.outcome for r in results]
    acc_ok = [r.recognition.decision.pid == it["pid"] for r, it in zip(results, items)
              if r.outcome == "accept"]
    return {"n": n, "pids": len(per),
            "top1": np.mean(top1) if n else 0, "top3": np.mean(top3) if n else 0,
            "macro_recall": np.mean([np.mean(v) for v in per.values()]) if per else 0,
            "hard_pair_acc": np.mean(hard) if hard else float("nan"), "hard_n": len(hard),
            "accept": oc.count("accept") / max(n, 1), "ask": oc.count("ask") / max(n, 1),
            "unknown": oc.count("unknown") / max(n, 1),
            "accept_precision": np.mean(acc_ok) if acc_ok else float("nan"),
            "p95_ms": float(np.percentile(ms, 95)) if ms else 0.0}


def hard_pair_pids(rec: ClickRecognizer, pids) -> set:
    """Return PIDs that have a near-identical sibling in the gallery.

    That means the same look-alike group, or the same numbered series
    ("Thermo 2" / "Thermo 3").
    """
    by = defaultdict(set)
    for p in pids:
        if rec.groups.get(p):
            by["g:" + rec.groups[p]].add(p)
        by["s:" + series_base(rec.name(p))].add(p)
    return {p for s in by.values() if len(s) > 1 for p in s}


def contact_sheet(items, results, gallery: Gallery, path: Path, n: int = 20, cell: int = 200):
    fails = [(r.recognition.candidates[0].score if r.recognition.candidates else 0, it, r)
             for it, r in zip(items, results)
             if not r.recognition.candidates or r.recognition.candidates[0].pid != it["pid"]]
    fails.sort(key=lambda x: -x[0])          # most confident mistakes first
    fails = fails[:n]
    if not fails:
        return None

    def ref_img(pid):
        for m in gallery.meta:
            if m["pid"] == pid and m["source"] in ("catalog", "real") and m["path"] and \
                    Path(m["path"]).is_file():
                return cv2.imdecode(np.fromfile(m["path"], np.uint8), cv2.IMREAD_COLOR)
        return None

    def fit(img, label):
        tile = np.full((cell + 22, cell, 3), 255, np.uint8)
        if img is not None:
            h, w = img.shape[:2]
            f = cell / max(h, w)
            im = cv2.resize(img, (max(int(w * f), 1), max(int(h * f), 1)))
            tile[:im.shape[0], :im.shape[1]] = im
        cv2.putText(tile, label[:28], (3, cell + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        return tile

    rows = []
    for s, it, r in fails:
        wrong = r.recognition.candidates[0].pid if r.recognition.candidates else "-"
        rows.append(np.hstack([fit(it["crop"], f"query {it['pid']}"),
                               fit(ref_img(wrong), f"top1 {wrong} {s:.2f}"),
                               fit(ref_img(it["pid"]), f"correct {it['pid']}")]))
    sheet = np.vstack(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)
    return path


# ---- Main ----
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos")
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--model", default="models/dinov2s_reg_224.onnx")
    ap.add_argument("--exp", default="E?", help="experiment id for results.csv (E1..E7)")
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sources", default="catalog,augmented,real",
                    help="gallery sources to use (E1-E4: catalog; E5: catalog,augmented)")
    ap.add_argument("--add-train", action="store_true",
                    help="append the train split as 'real' gallery vectors (E6)")
    ap.add_argument("--val", type=float, default=0.3)
    ap.add_argument("--test", type=float, default=0.3)
    ap.add_argument("--open-set", type=float, default=0.10)
    ap.add_argument("--precision", type=float, default=0.98)
    ap.add_argument("--write-thresholds", action="store_true",
                    help="save the val-calibrated thresholds to <gallery>/thresholds.json")
    ap.add_argument("--results", type=Path, default=Path(__file__).with_name("results.csv"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    data = click_photos(a.photos)
    if not data:
        print("no labelled Click photos (PID in filename or labels.csv kind=click)")
        return 1
    for d in data:
        d["split"] = split_of(d["session"], a.val, a.test)
    if len({d["split"] for d in data}) < 2:
        print("[split] one capture session only - splitting by photo instead (less honest)")
        for d in data:
            d["split"] = split_of(d["photo"].sha1, a.val, a.test)

    emb = get_embedder(a.model)
    gal = Gallery.load(a.gallery, emb.model_version, emb.preprocess_version)
    gal = gal.only_sources(a.sources.split(","))
    base = ClickConfig(tta=a.tta, ocr=a.ocr, verify=a.verify)
    rec = ClickRecognizer(emb, gal, a.backend if a.ocr else None, Catalog.load(a.catalog), base)

    if a.add_train:
        tr = [d for d in data if d["split"] == "train"]
        crops = [crop_box(d["photo"].image, d["box"]) if d["box"] else d["photo"].image
                 for d in tr]
        for d, v in zip(tr, emb.embed(crops) if crops else []):
            gal.add(d["pid"], v[None], "real", [d["photo"].name],
                    [str(d["photo"].path or "")])
        rec.gallery = gal
        print(f"[gallery] + {len(tr)} real train photos")

    # compute one descriptor and one OCR read per crop; everything else reuses them
    items = []
    lat = []
    for d in data:
        if d["split"] == "train":
            continue
        img = d["photo"].image
        crop = crop_box(img, d["box"], base.box_expand) if d["box"] else img
        t0 = time.perf_counter()
        q = rec.describe(crop, a.tta)
        o = ocr_click(crop, rec.backend) if (a.ocr and rec.backend) else None
        lat.append((time.perf_counter() - t0) * 1000)
        items.append({**d, "image": img, "crop": crop, "q": q, "ocr": o})
    val = [i for i in items if i["split"] == "val"]
    test = [i for i in items if i["split"] == "test"]
    print(f"[split] train {sum(d['split'] == 'train' for d in data)} · val {len(val)} · "
          f"test {len(test)} photos")

    cfg = calibrate(rec, val, base, a.precision) if val else base
    print(f"[calibrate] T_accept {cfg.t_accept:.2f} · T_unknown {cfg.t_unknown:.2f} · "
          f"M {cfg.margin:.2f}  (from val)")
    if a.write_thresholds:
        import json
        p = Path(a.gallery) / "thresholds.json"
        p.write_text(json.dumps({"t_accept": cfg.t_accept, "t_unknown": cfg.t_unknown,
                                 "margin": cfg.margin, "calibrated_on": "val",
                                 "precision_target": a.precision, "exp": a.exp,
                                 "model_version": emb.model_version}, indent=1),
                     encoding="utf-8")
        print(f"[calibrate] wrote {p}")

    report = test or val
    split_name = "test" if test else "val (no test split!)"
    res = outcomes(rec, report, cfg)
    hard = hard_pair_pids(rec, gal.pids)
    m = metrics(rec, report, res, hard, lat)

    # open set: remove a fixed 10 % of PIDs from the gallery and see where their photos end
    # up
    rng = np.random.default_rng(a.seed)
    pids = sorted({i["pid"] for i in report})
    k = max(1, int(round(len(pids) * a.open_set))) if pids else 0
    gone = set(rng.choice(pids, k, replace=False)) if k else set()
    small = gal.without_pids(gone)
    os_items = [{**i, "gallery": small} for i in report if i["pid"] in gone]
    os_res = outcomes(rec, os_items, cfg) if os_items else []
    m["open_set_unknown"] = np.mean([r.outcome == "unknown" for r in os_res]) if os_res else float("nan")
    m["open_set_false_accept"] = np.mean([r.outcome == "accept" for r in os_res]) if os_res else float("nan")

    sheet = contact_sheet(report, res, gal, Path(__file__).with_name(f"failures_{a.exp}.jpg"))
    sh = hashlib.sha1("".join(sorted(i["photo"].sha1 for i in test)).encode()).hexdigest()[:12]
    row = {"date": time.strftime("%Y-%m-%d %H:%M"), "exp": a.exp,
           "model_version": emb.model_version, "preprocess_version": emb.preprocess_version,
           "gallery": a.sources + ("+train" if a.add_train else ""),
           "rerank": "+".join(x for x, on in (("tta", a.tta), ("ocr", a.ocr),
                                               ("verify", a.verify)) if on) or "none",
           "split": split_name, "t_accept": cfg.t_accept, "t_unknown": cfg.t_unknown,
           "margin": cfg.margin, "split_hash": sh, **m}
    new = not a.results.is_file()
    with open(a.results, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, RESULTS_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k2: (round(v, 4) if isinstance(v, (float, np.floating)) else v)
                    for k2, v in row.items()})

    pct = lambda v: "—" if v != v else f"{100 * v:.1f}%"   # noqa: E731
    print(f"\n{a.exp} · {emb.model_version} · gallery {row['gallery']} · re-rank {row['rerank']}"
          f" · {split_name}: {m['n']} photos, {m['pids']} Clicks")
    print(f"  top-1 {pct(m['top1'])} · top-3 {pct(m['top3'])} · macro recall "
          f"{pct(m['macro_recall'])} · hard pairs {pct(m['hard_pair_acc'])} (n={m['hard_n']})")
    print(f"  accept {pct(m['accept'])} (precision {pct(m['accept_precision'])}) · ask "
          f"{pct(m['ask'])} · unknown {pct(m['unknown'])}")
    print(f"  open set ({len(gone)} PIDs removed): unknown {pct(m['open_set_unknown'])} · "
          f"false accept {pct(m['open_set_false_accept'])} · p95 {m['p95_ms']:.0f} ms/crop")
    print(f"  -> {a.results}" + (f", worst failures: {sheet}" if sheet else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
