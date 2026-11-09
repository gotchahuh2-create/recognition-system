"""Experiment Management: test a hypothesis by running variants against a baseline.

An experiment lives in ``docs/experiments/E-NNN-slug/``:

    spec.yaml    what is tested (written by a person or by Claude, reviewed)
    results.json gate evidence per variant (written by ``run``)
    report.md    readable summary (written by ``run``)

spec.yaml::

    id: E-001
    title: Geometry-first card localisation
    hypothesis: H-002                 # docs/hypotheses/H-002-*.md
    datasets: [syn_cards, real3, captures]
    baseline: active                  # the promoted config, or a dict like a variant
    baseline_report: <run_id>         # optional: reuse an existing bench report
    variants:
      - name: geometry_first
        stages: {card_localise: geometry_first}
        params: {}
    gates: {tolerance: 0, latency_budget: 1.25}

Variants may only choose registered implementations and override parameters
listed in ``config/params.yaml``: no code is generated or modified here.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from . import bench
from .stages import load_active, resolve
from .util import BENCH_DIR, DOCS_DIR, read_json, write_json

EXP_DIR = DOCS_DIR / "experiments"


def find(exp_id: str) -> Path:
    hits = sorted(EXP_DIR.glob(f"{exp_id}*"))
    if not hits:
        raise FileNotFoundError(f"no experiment folder for {exp_id} in {EXP_DIR}")
    return hits[0]


def load_spec(path: Path) -> dict:
    return yaml.safe_load((path / "spec.yaml").read_text(encoding="utf-8"))


def _cfg(spec_cfg, name):
    if spec_cfg in (None, "active"):
        rc = load_active()
        rc.name = name
        return rc
    return resolve({"name": name, **spec_cfg})


def run(exp_id: str, progress=print) -> dict:
    folder = find(exp_id)
    spec = load_spec(folder)
    ds = spec["datasets"]
    g = spec.get("gates", {})
    if spec.get("baseline_report"):
        base = read_json(BENCH_DIR / "reports" / f"{spec['baseline_report']}.json")
    else:
        base = bench.run(ds, _cfg(spec.get("baseline"), f"{spec['id']}-baseline"), progress=progress)
    results = {"id": spec["id"], "hypothesis": spec.get("hypothesis"), "title": spec["title"],
               "baseline": base["run_id"], "variants": []}
    for v in spec["variants"]:
        base_rc = _cfg(spec.get("baseline"), "tmp")
        cfg = {"name": f"{spec['id']}-{v['name']}",
               "stages": {**base_rc.stages, **(v.get("stages") or {})},
               "params": {**base_rc.params, **(v.get("params") or {})}}
        rc = resolve(cfg)
        if v.get("replay"):
            # post-hoc stages (guards) only read final decisions: replay them on the baseline
            rep = _replay(base, rc, v["replay"])
        else:
            rep = bench.run(ds, rc, progress=progress)
        cmp = bench.compare(rep, base, tolerance=g.get("tolerance", 0),
                            latency_budget=g.get("latency_budget", 1.25))
        passed = all(cmp["gates"][k] for k in ("no_new_wrong", "no_lost_correct",
                                                "latency_within_budget"))
        improved = (len(cmp["new_correct"]) > len(cmp["lost_correct"])
                    or (cmp["fixed_wrong"] and not cmp["lost_correct"]))
        faster = verdict_faster(cmp)
        results["variants"].append({"name": v["name"], "config": rep["config"],
                                    "report": rep["run_id"], "compare": cmp,
                                    "passed_gates": passed, "improved": bool(improved or faster),
                                    "faster_same_answers": faster,
                                    "summary": rep["datasets"]})
    write_json(folder / "results.json", results)
    (folder / "report.md").write_text(to_markdown(results, base), encoding="utf-8")
    return results


SPEEDUP_MIN = 0.9  # an exact acceleration must save at least 10 % of total processing time


def verdict_faster(cmp: dict) -> bool:
    r = (cmp.get("total_time") or {}).get("ratio")
    return bool(cmp["gates"]["identical_decisions"] and r is not None and r <= SPEEDUP_MIN)


def reassess(exp_id: str) -> dict:
    """Recompute verdicts of a finished experiment from its stored reports (no re-run)."""
    folder = find(exp_id)
    res = read_json(folder / "results.json")
    spec = load_spec(folder)
    g = spec.get("gates", {})
    base = read_json(BENCH_DIR / "reports" / f"{res['baseline']}.json")
    for v in res["variants"]:
        rep = read_json(BENCH_DIR / "reports" / f"{v['report']}.json")
        cmp = bench.compare(rep, base, tolerance=g.get("tolerance", 0),
                            latency_budget=g.get("latency_budget", 1.25))
        passed = all(cmp["gates"][k] for k in ("no_new_wrong", "no_lost_correct",
                                                "latency_within_budget"))
        improved = (len(cmp["new_correct"]) > len(cmp["lost_correct"])
                    or (cmp["fixed_wrong"] and not cmp["lost_correct"]))
        faster = verdict_faster(cmp)
        v.update({"compare": cmp, "passed_gates": passed, "improved": bool(improved or faster),
                  "faster_same_answers": faster})
    write_json(folder / "results.json", res)
    (folder / "report.md").write_text(to_markdown(res, base), encoding="utf-8")
    return res


def to_markdown(res: dict, base: dict) -> str:
    L = [f"# {res['id']}: {res['title']}", "",
         f"Hypothesis: {res.get('hypothesis') or '–'} · baseline report `{res['baseline']}`", ""]
    for v in res["variants"]:
        c = v["compare"]
        L += [f"## Variant `{v['name']}` — report `{v['report']}`", "",
              f"- gates: " + ", ".join(f"{k}={'pass' if ok else 'FAIL'}" for k, ok in c["gates"].items()
                                     if k != "identical_decisions")
              + f" · decisions identical to baseline: {'yes' if c['gates']['identical_decisions'] else 'no'}",
              f"- new correct: {len(c['new_correct'])} · lost correct: {len(c['lost_correct'])} · "
              f"new wrong: {len(c['new_wrong'])} · wrong fixed: {len(c.get('fixed_wrong', []))} · "
              f"decisions changed: {len(c['changed'])}",
              f"- total processing time: {c.get('total_time', {}).get('base_s')} s -> "
              f"{c.get('total_time', {}).get('cand_s')} s (x{c.get('total_time', {}).get('ratio')})",
              f"- verdict: {'passes gates and improves' if v['passed_gates'] and v['improved'] else 'passes gates, no improvement' if v['passed_gates'] else 'fails gates'}"
              + (" (same answers, faster)" if v.get('faster_same_answers') else ""),
              ""]
        for title, lst in (("New correct", c["new_correct"]), ("Lost correct", c["lost_correct"]),
                           ("New wrong", c["new_wrong"]), ("Wrong answers fixed", c.get("fixed_wrong", []))):
            if lst:
                L += [f"{title}:", ""] + [f"- `{x}`" for x in lst[:20]] + [""]
        L += ["| dataset | base p95 ms | variant p95 ms | ratio |", "|---|---|---|---|"]
        for dn, l in c["latency"].items():
            L.append(f"| {dn} | {l['base_p95']} | {l['cand_p95']} | {l['ratio']} |")
        L.append("")
    return "\n".join(L)


def _replay(base: dict, rc, what: str) -> dict:
    from . import monitor
    from .util import RUNS_DIR, timestamp, git_commit
    if what != "guards":
        raise ValueError(f"unknown replay kind {what!r}")
    from vision.catalog import Catalog
    from .guards import replay
    rep = replay(base, Catalog.load())
    rep["run_id"] = f"{timestamp()}-{git_commit().replace('+', '_')}-{rc.hash}-replay"
    rep["config"] = rc.as_dict()
    rep["replayed_from"] = base["run_id"]
    traces = monitor.load(RUNS_DIR / "traces" / f"{base['run_id']}.jsonl")
    for t in traces:  # the replayed decisions, so rescoring sees them
        k = f"{t['dataset']}/{t['item']}"
        t["decisions_flat"] = rep["items"][k]["decisions"]
    rep = bench.rescore(rep, traces)
    # latency: identical to the baseline by construction (a guard costs microseconds)
    for dn in rep["datasets"]:
        rep["datasets"][dn]["latency_ms"] = base["datasets"][dn]["latency_ms"]
    bench.save(rep)
    return rep
