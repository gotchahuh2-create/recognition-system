"""Persistent Engineering Memory: the files a new session reads to know where things stand.

Everything is plain markdown / YAML / JSON in the repository:

    docs/STATE.md          generated summary (``python -m phoenix state``)
    docs/adr/              decisions (migrated DECISIONS.md + PHOENIX ADRs from 0100)
    docs/hypotheses/       H-NNN-*.md, front matter: id, status, weakness, experiment
    docs/experiments/      E-NNN-*/spec.yaml, results.json, report.md
    docs/proposals/        P-NNN.json/.md
    docs/weaknesses.md     generated register (``python -m phoenix diagnose``)
    bench/baselines.yaml   which bench report is the reference for each config version

This module only reads them and writes STATE.md.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import yaml

from .util import BENCH_DIR, CONFIG_DIR, DOCS_DIR, git_commit, read_json

FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)


def front_matter(path: Path) -> dict:
    m = FRONT.match(path.read_text(encoding="utf-8"))
    return yaml.safe_load(m.group(1)) if m else {}


def hypotheses() -> list[dict]:
    return [{"file": p.name, **front_matter(p)}
            for p in sorted((DOCS_DIR / "hypotheses").glob("H-*.md"))]


def experiments() -> list[dict]:
    out = []
    for d in sorted((DOCS_DIR / "experiments").glob("E-*")):
        spec = yaml.safe_load((d / "spec.yaml").read_text(encoding="utf-8")) if (d / "spec.yaml").exists() else {}
        res = read_json(d / "results.json") if (d / "results.json").exists() else None
        verdicts = ([f"{v['name']}: " + ("passes+improves" if v["passed_gates"] and v["improved"]
                                          else "passes" if v["passed_gates"] else "fails")
                     for v in res["variants"]] if res else ["not run"])
        out.append({"id": spec.get("id", d.name), "title": spec.get("title", ""),
                    "hypothesis": spec.get("hypothesis"), "verdicts": verdicts})
    return out


def proposals() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((DOCS_DIR / "proposals").glob("P-*.json"))]


def baselines() -> dict:
    p = BENCH_DIR / "baselines.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_state() -> Path:
    from . import params as P
    act = yaml.safe_load((CONFIG_DIR / "active.yaml").read_text(encoding="utf-8"))
    reg = P.load_params()["params"]
    bl = baselines()
    ref = bl.get(act["active"])
    L = ["# PHOENIX state", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python -m phoenix state` at git `{git_commit()}`. "
         "Read this first; details are linked.", "",
         "## Configuration", "",
         f"- Active config: **{act['active']}** (last known good: {act['last_known_good']}); "
         "history in `config/active.yaml`.",
         f"- Parameters: {len(reg)} in `config/params.yaml` "
         f"({sum(1 for v in reg.values() if v.get('bound') and not v.get('via'))} bound, need a code change to tune).",
         "- Stage choices and their status: `config/registry.yaml`.", ""]
    if ref:
        rep = read_json(BENCH_DIR / "reports" / f"{ref}.json")
        L += ["## Reference benchmark", "", f"Report `{ref}` (backend `{rep['backend']}`).", "",
              "| dataset | field | correct | wrong | abstain | n |", "|---|---|---|---|---|---|"]
        for dn, d in rep["datasets"].items():
            for f, s in d["fields"].items():
                extra = (f" (+{s['unverified']} unverified, {s['agree_provisional']} agree)"
                         if s.get("unverified") else "")
                L.append(f"| {dn} | {f} | {s['correct']} | {s['wrong']} | {s['abstain']} | {s['n']}{extra} |")
        L.append("")
    L += ["## Hypotheses", "", "| id | status | title | weakness | experiment |", "|---|---|---|---|---|"]
    for h in hypotheses():
        L.append(f"| {h.get('id')} | {h.get('status')} | {h.get('title')} | {h.get('weakness', '–')} | {h.get('experiment', '–')} |")
    L += ["", "## Experiments", ""]
    for e in experiments():
        L.append(f"- **{e['id']}** {e['title']} ({e['hypothesis']}): " + "; ".join(e["verdicts"]))
    L += ["", "## Proposals", ""]
    props = proposals()
    L += [f"- **{p['id']}** {p['status']}: {p['change']} (from {p['experiment']})" for p in props] or ["- none"]
    L += ["", "## Where to look", "",
          "- Weaknesses: `docs/weaknesses.md` · Decisions: `docs/adr/README.md` · "
          "Architecture: `docs/architecture/README.md` · Efficiency: `docs/reports/efficiency.md`",
          "- Rules for working here: `CLAUDE.md`", ""]
    out = DOCS_DIR / "STATE.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out
