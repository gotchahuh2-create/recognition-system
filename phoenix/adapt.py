"""Adaptation Engine: turn experiment evidence into change *proposals*. It never applies them.

A proposal is the smallest configuration change a passing experiment
supports: a different registered implementation for a stage and/or parameter
overrides. It records the evidence (experiment, bench reports, gate results)
and the risks, and waits in ``docs/proposals/P-NNN.json`` with status
``pending``. Only ``phoenix.validate.promote`` (run by a person, with their
name) can turn an ``approved`` proposal into a new config version.

What this module can never do: write code, edit ``vision/``, change a
``protected`` stage to anything but another registered implementation, or
change a ``bound`` parameter (that needs a code change and an ADR).
"""

from __future__ import annotations

import time
from pathlib import Path

from .stages import load_active, load_registry
from .util import DOCS_DIR, read_json, write_json

PROP_DIR = DOCS_DIR / "proposals"


def _next_id() -> str:
    PROP_DIR.mkdir(parents=True, exist_ok=True)
    n = [int(p.stem.split("-")[1]) for p in PROP_DIR.glob("P-*.json")]
    return f"P-{(max(n) + 1 if n else 1):03d}"


def propose_from_experiment(results: dict) -> list[Path]:
    """One proposal per variant that passed every gate AND improved the baseline."""
    act = load_active()
    reg = load_registry()
    out = []
    for v in results["variants"]:
        if not (v["passed_gates"] and v["improved"]):
            continue
        st = {k: val for k, val in v["config"]["stages"].items() if act.stages.get(k) != val}
        pa = {k: val for k, val in v["config"]["params"].items() if act.params.get(k) != val}
        risks = []
        for k, val in st.items():
            impl = {str(i): d for i, d in reg["stages"][k]["impls"].items()}[val]
            if impl.get("status") == "experimental":
                risks.append(f"{k}={val} is experimental: promotion also changes its status to stable")
            prot = {str(i): d for i, d in reg["stages"][k]["impls"].items()}[act.stages[k]]
            if prot.get("status") == "protected":
                risks.append(f"replaces protected default {k}={act.stages[k]} as the DEFAULT "
                             f"(the protected code stays in place and selectable)")
        c = v["compare"]
        unverified = [k for k in c["changed"]
                      if not any(f in ("board", "mcu", "mcu_card", "mcu_card_layout")
                                 for f in _truth_fields(k))]
        if unverified:
            risks.append(f"{len(unverified)} decision(s) changed on photos without ground truth "
                         f"(cannot be judged until labelled): " + ", ".join(unverified[:5]))
        pid = _next_id()
        prop = {"id": pid, "status": "pending", "created": time.strftime("%Y-%m-%d %H:%M"),
                "experiment": results["id"], "hypothesis": results.get("hypothesis"),
                "change": {"stages": st, "params": pa}, "from_version": act.name,
                "evidence": {"baseline_report": results["baseline"], "variant_report": v["report"],
                             "gates": c["gates"], "new_correct": c["new_correct"],
                             "lost_correct": c["lost_correct"], "new_wrong": c["new_wrong"],
                             "fixed_wrong": c.get("fixed_wrong", []),
                             "identical_decisions": c["gates"]["identical_decisions"],
                             "latency": c["latency"]},
                "risks": risks, "approved_by": None, "decided": None}
        p = write_json(PROP_DIR / f"{pid}.json", prop)
        p.with_suffix(".md").write_text(to_markdown(prop), encoding="utf-8")
        out.append(p)
    return out


def to_markdown(p: dict) -> str:
    e = p["evidence"]
    return "\n".join([
        f"# {p['id']} — {p['status'].upper()}", "",
        f"From experiment {p['experiment']} (hypothesis {p['hypothesis']}), against config `{p['from_version']}`.", "",
        "## Change", "", f"- stages: {p['change']['stages'] or 'none'}",
        f"- params: {p['change']['params'] or 'none'}", "",
        "## Evidence", "",
        f"- reports: baseline `{e['baseline_report']}`, variant `{e['variant_report']}`",
        "- gates: " + ", ".join(f"{k}={'pass' if v else 'FAIL'}" for k, v in e["gates"].items()
                                if k != "identical_decisions"),
        f"- new correct ({len(e['new_correct'])}): " + ", ".join(e["new_correct"][:10]),
        f"- lost correct: {len(e['lost_correct'])} · new wrong: {len(e['new_wrong'])} · "
        f"wrong answers fixed: {len(e.get('fixed_wrong', []))}", "",
        "## Risks", "", *([f"- {r}" for r in p["risks"]] or ["- none found"]), "",
        "## Decision", "",
        "Approve: `python -m phoenix promote " + p["id"] + " --approved-by \"<your name>\"`  ",
        "Reject: `python -m phoenix reject " + p["id"] + " --by \"<your name>\" --why \"...\"`", ""])


def load(pid: str) -> dict:
    return read_json(PROP_DIR / f"{pid}.json")


def save(p: dict) -> Path:
    path = write_json(PROP_DIR / f"{p['id']}.json", p)
    path.with_suffix(".md").write_text(to_markdown(p), encoding="utf-8")
    return path


def _truth_fields(item_key: str) -> list[str]:
    from .datasets import Dataset
    dn, f = item_key.split("/", 1)
    try:
        it = next(i for i in Dataset.load(dn).m["items"] if i["file"] == f)
    except (StopIteration, FileNotFoundError):
        return []
    return list(it.get("truth", {}))
