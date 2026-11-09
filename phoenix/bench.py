"""Algorithm Evaluation: run one configuration over benchmark datasets, score, report, compare.

    python -m phoenix bench --datasets syn_ocr,syn_cards,real3,captures [--config FILE] [--name X]

One harness and one metric set for every stage (the ``eval/*.py`` scripts stay
for their specialised studies). Per dataset and field:

* ``correct`` / ``wrong`` / ``abstain`` (abstain is never wrong; for truth
  ``"none"`` the only correct answer is to abstain);
* ``precision`` = correct / (correct + wrong) when answering, ``coverage`` =
  answered / scored;
* ``top3`` = truth among the top-3 candidates;
* ``agree`` = decision equals the provisional label (unverified data);
* latency p50 / p95 / max and mean time per stage.

Every run gets a run id (time + git commit + config hash), its traces
(``runs/traces/``) and a report (``bench/reports/<run_id>.json`` + ``.md``).
``compare`` turns two reports into gate results (``phoenix.validate``).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import monitor
from .datasets import FIELDS, Dataset
from .stages import PhoenixPipeline, RunConfig, resolve
from .util import BENCH_DIR, git_commit, rel, timestamp, write_json

REPORT_DIR = BENCH_DIR / "reports"


def outcome(field: str, trace: dict, truth) -> str:
    pred = trace["decisions_flat"].get(field)
    if truth == "none":
        return "correct" if pred is None else "wrong"
    if pred is None:
        return "abstain"
    if field == "mcu":
        from vision.textmatch import compact
        return "correct" if compact(pred) == compact(truth) else "wrong"
    return "correct" if pred == truth else "wrong"


def _in_top(field: str, truth: str, top: list) -> bool:
    if field == "mcu":
        from vision.textmatch import compact
        return compact(truth) in {compact(t) for t in top}
    return truth in top


def score_item(trace: dict, truth: dict, provisional: dict) -> dict:
    out = {}
    for f in FIELDS:
        if f in truth:
            o = outcome(f, trace, truth[f])
            top = trace["top3"].get(f)  # no candidate list for mcu_card_layout -> not scored
            out[f] = {"outcome": o, "truth": truth[f], "pred": trace["decisions_flat"].get(f),
                      "top3": (_in_top(f, truth[f], top)
                               if top is not None and truth[f] != "none" else None)}
        elif f in provisional:
            out[f] = {"outcome": "unverified", "provisional": provisional[f],
                      "pred": trace["decisions_flat"].get(f),
                      "agree": trace["decisions_flat"].get(f) == provisional[f]}
    return out


def score_sockets(trace: dict, truth: dict) -> dict:
    """truth: {"mikroBUS 1": "empty"|"occupied", ...}; unknown = abstain."""
    got = {s["slot"]: s["occupancy"] for s in trace.get("sockets", [])}
    out = {}
    for slot, t in truth.items():
        g = got.get(slot)
        out[slot] = ("abstain" if g in (None, "unknown") else "correct" if g == t else "wrong")
    return out


def summarise(rows: list[dict]) -> dict:
    """rows: [{"item", "scores": {field: {...}}, "wall_ms", "timings_ms"}] for one dataset."""
    fields = {}
    for f in FIELDS:
        sc = [r["scores"][f] for r in rows if f in r["scores"]]
        if not sc:
            continue
        ver = [s for s in sc if s["outcome"] != "unverified"]
        c = sum(s["outcome"] == "correct" for s in ver)
        w = sum(s["outcome"] == "wrong" for s in ver)
        a = sum(s["outcome"] == "abstain" for s in ver)
        t3 = [s["top3"] for s in ver if s.get("top3") is not None]
        unv = [s for s in sc if s["outcome"] == "unverified"]
        fields[f] = {"n": len(ver), "correct": c, "wrong": w, "abstain": a,
                     "precision": round(c / (c + w), 3) if c + w else None,
                     "coverage": round((c + w) / len(ver), 3) if ver else None,
                     "top3": f"{sum(t3)}/{len(t3)}" if t3 else None}
        if unv:
            fields[f]["unverified"] = len(unv)
            fields[f]["agree_provisional"] = sum(bool(s["agree"]) for s in unv)
    so = [o for r in rows for o in (r.get("sockets") or {}).values()]
    if so:
        c, w, a = so.count("correct"), so.count("wrong"), so.count("abstain")
        fields["sockets"] = {"n": len(so), "correct": c, "wrong": w, "abstain": a,
                             "precision": round(c / (c + w), 3) if c + w else None,
                             "coverage": round((c + w) / len(so), 3), "top3": None}
    wall = np.array([r["wall_ms"] for r in rows], float)
    stages = {}
    for r in rows:
        for k, v in r["timings_ms"].items():
            if isinstance(v, (int, float)) and k != "total":
                stages.setdefault(k, []).append(v)
    return {"n": len(rows), "fields": fields,
            "latency_ms": {"p50": round(float(np.percentile(wall, 50))),
                           "p95": round(float(np.percentile(wall, 95))),
                           "max": round(float(wall.max())), "mean": round(float(wall.mean()))},
            "stage_mean_ms": {k: round(float(np.mean(v))) for k, v in stages.items()}}


def run(datasets: list[str], cfg: dict | RunConfig | None = None, name: str = "",
        write: bool = True, profile: bool = False, progress=print) -> dict:
    rc = cfg if isinstance(cfg, RunConfig) else resolve(cfg)
    if name:
        rc.name = name
    commit = git_commit()
    run_id = f"{timestamp()}-{commit.replace('+', '_')}-{rc.hash}"
    pp = PhoenixPipeline(rc)
    from vision.loader import load_photo
    report = {"run_id": run_id, "git": commit, "config": rc.as_dict(),
              "backend": pp.pipeline.backend.name, "datasets": {}, "items": {}}
    prof = None
    if profile:
        import cProfile
        prof = cProfile.Profile()
    t_all = time.perf_counter()
    for dn in datasets:
        ds = Dataset.load(dn)
        rows, traces = [], []
        for it in ds.items():
            photo = load_photo(it.path)
            t0 = time.perf_counter()
            if prof:
                prof.enable()
            res = pp.run(photo)
            if prof:
                prof.disable()
            wall = (time.perf_counter() - t0) * 1000
            tr = monitor.trace_of(res, run_id=run_id, item=it.file, dataset=ds.name,
                                  config={"name": rc.name, "hash": rc.hash}, wall_ms=wall)
            traces.append(tr)
            sc = score_item(tr, it.truth, it.provisional)
            so = score_sockets(tr, it.truth.get("sockets", {})) if tr["sockets"] else (
                {k: "abstain" for k in it.truth.get("sockets", {})})
            rows.append({"item": it.file, "scores": sc, "wall_ms": wall,
                         "timings_ms": res.timings_ms, "sockets": so})
            report["items"][f"{ds.name}/{it.file}"] = {
                "decisions": tr["decisions_flat"], "scores": sc, "wall_ms": round(wall),
                "reasons": tr["reasons"], "mcu_source": tr["mcu_source"], "card": tr["card"],
                "sockets": tr["sockets"], "socket_scores": so}
            if progress:
                progress(f"  {ds.name}/{it.file}: {round(wall)} ms "
                         + " ".join(f"{f}={s['outcome']}" for f, s in sc.items()))
        report["datasets"][ds.name] = summarise(rows)
        if write:
            monitor.append(run_id, traces)
    report["total_s"] = round(time.perf_counter() - t_all, 1)
    from . import accel
    if accel.stats():
        report["accel"] = accel.stats()
    if prof:
        report["_profile"] = prof
    if write:
        save(report)
    return report


def rescore(report: dict, traces: list[dict]) -> dict:
    """Re-score a stored run against the current manifests (no recognition re-run)."""
    by = {f"{t['dataset']}/{t['item']}": t for t in traces}
    per_ds: dict[str, list] = {}
    for dn in report["datasets"]:
        ds = {it.file: it for it in Dataset.load(dn).items()}
        rows = []
        for key, item in report["items"].items():
            d, f = key.split("/", 1)
            if d != dn or key not in by:
                continue
            sc = score_item(by[key], ds[f].truth, ds[f].provisional)
            item["scores"] = sc
            st = ds[f].truth.get("sockets", {})
            so = score_sockets(by[key], st) if by[key].get("sockets") else {k: "abstain" for k in st}
            item["socket_scores"] = so
            rows.append({"item": f, "scores": sc, "wall_ms": item["wall_ms"],
                         "timings_ms": by[key]["timings_ms"], "sockets": so})
        per_ds[dn] = summarise(rows)
    report["datasets"] = per_ds
    return report


def save(report: dict) -> Path:
    rep = {k: v for k, v in report.items() if not k.startswith("_")}
    p = write_json(REPORT_DIR / f"{report['run_id']}.json", rep)
    p.with_suffix(".md").write_text(to_markdown(rep), encoding="utf-8")
    return p


def to_markdown(r: dict) -> str:
    L = [f"# Bench report {r['run_id']}", "",
         f"- config: `{r['config']['name']}` ({r['config']['hash']}), backend `{r['backend']}`, "
         f"git `{r['git']}`, total {r['total_s']} s",
         f"- stages: " + ", ".join(f"{k}={v}" for k, v in r["config"]["stages"].items()),
         f"- param overrides: {r['config']['params'] or 'none'}", "",
         "| dataset | field | n | correct | wrong | abstain | precision | coverage | top-3 | provisional agree |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for dn, d in r["datasets"].items():
        for f, s in d["fields"].items():
            agree = (f"{s['agree_provisional']}/{s['unverified']}" if s.get("unverified") else "")
            L.append(f"| {dn} | {f} | {s['n']} | {s['correct']} | {s['wrong']} | {s['abstain']} | "
                     f"{s['precision'] if s['precision'] is not None else '–'} | "
                     f"{s['coverage'] if s['coverage'] is not None else '–'} | {s['top3'] or '–'} | {agree} |")
    L += ["", "| dataset | n | p50 ms | p95 ms | max ms | stage means (ms) |", "|---|---|---|---|---|---|"]
    for dn, d in r["datasets"].items():
        lat = d["latency_ms"]
        L.append(f"| {dn} | {d['n']} | {lat['p50']} | {lat['p95']} | {lat['max']} | "
                 + ", ".join(f"{k} {v}" for k, v in d["stage_mean_ms"].items()) + " |")
    return "\n".join(L) + "\n"


def decisions(report: dict) -> dict:
    return {k: v["decisions"] for k, v in report["items"].items()}


def compare(cand: dict, base: dict, tolerance: int = 0, latency_budget: float = 1.25) -> dict:
    """Gate evidence: what changed between a baseline report and a candidate report."""
    db, dc = decisions(base), decisions(cand)
    common = sorted(set(db) & set(dc))
    changed = {k: {"base": db[k], "cand": dc[k]} for k in common if db[k] != dc[k]}
    new_wrong, new_correct, lost_correct, fixed_wrong = [], [], [], []
    for k in common:
        sb, sc = base["items"][k]["scores"], cand["items"][k]["scores"]
        pairs = [(f, sb.get(f, {}).get("outcome"), sc[f].get("outcome")) for f in sc]
        bs, cs = base["items"][k].get("socket_scores") or {}, cand["items"][k].get("socket_scores") or {}
        pairs += [(f"socket:{s}", bs.get(s), cs[s]) for s in cs]
        for f, ob, oc in pairs:
            if oc == "wrong" and ob != "wrong":
                new_wrong.append(f"{k}:{f}")
            if oc == "correct" and ob != "correct":
                new_correct.append(f"{k}:{f}")
            if ob == "correct" and oc != "correct":
                lost_correct.append(f"{k}:{f}")
            if ob == "wrong" and oc != "wrong":
                fixed_wrong.append(f"{k}:{f}")
    lat = {}
    for dn in set(base["datasets"]) & set(cand["datasets"]):
        b, c = base["datasets"][dn]["latency_ms"]["p95"], cand["datasets"][dn]["latency_ms"]["p95"]
        lat[dn] = {"base_p95": b, "cand_p95": c, "ratio": round(c / b, 3) if b else None}
    over = [dn for dn, v in lat.items() if v["ratio"] and v["ratio"] > latency_budget]
    # total processing time over the common datasets (mean x n), the measure for speed-ups
    tb = sum(base["datasets"][d]["latency_ms"]["mean"] * base["datasets"][d]["n"] for d in lat)
    tc = sum(cand["datasets"][d]["latency_ms"]["mean"] * cand["datasets"][d]["n"] for d in lat)
    total = {"base_s": round(tb / 1000, 1), "cand_s": round(tc / 1000, 1),
             "ratio": round(tc / tb, 3) if tb else None}
    gates = {
        "no_new_wrong": not new_wrong,
        "no_lost_correct": len(lost_correct) <= tolerance,
        "latency_within_budget": not over,
        "identical_decisions": not changed,
    }
    return {"base": base["run_id"], "cand": cand["run_id"], "items_compared": len(common),
            "gates": gates, "changed": changed, "new_wrong": new_wrong,
            "new_correct": new_correct, "lost_correct": lost_correct, "fixed_wrong": fixed_wrong,
            "latency": lat, "latency_over_budget": over, "total_time": total}


def print_summary(report: dict) -> None:
    print(to_markdown({k: v for k, v in report.items() if not k.startswith("_")}))
    print("report:", rel(REPORT_DIR / f"{report['run_id']}.json"))
