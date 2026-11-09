"""PHOENIX command line.

    python -m phoenix datasets build            # (re)write bench/datasets/*.json
    python -m phoenix datasets template         # photos/captures/labels.TEMPLATE.csv to fill in
    python -m phoenix params extract|check      # config/params.yaml from the code / drift check
    python -m phoenix bench --datasets syn_ocr,real3 [--config v1|FILE] [--name X] [--set-baseline]
    python -m phoenix gate BASE_REPORT CAND_REPORT
    python -m phoenix diagnose [--report RUN_ID]
    python -m phoenix experiment E-001
    python -m phoenix promote P-001 --approved-by "Name"
    python -m phoenix reject P-001 --by "Name" --why "..."
    python -m phoenix rollback [vN]
    python -m phoenix state                     # regenerate docs/STATE.md
    python -m phoenix ingest                    # capture sidecars -> runs/traces/captures.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _cfg(arg: str | None):
    from .stages import load_active, resolve
    from .util import CONFIG_DIR
    if not arg or arg == "active":
        return load_active()
    p = Path(arg)
    if not p.exists():
        p = CONFIG_DIR / "versions" / f"{arg}.yaml"
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    return resolve({"name": d.get("name", p.stem), **d})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phoenix", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("datasets"); d.add_argument("action", choices=["build", "template"])
    p = sub.add_parser("params"); p.add_argument("action", choices=["extract", "check"])
    b = sub.add_parser("bench")
    b.add_argument("--datasets", default="syn_ocr,syn_cards,real3,captures")
    b.add_argument("--config"); b.add_argument("--name", default="")
    b.add_argument("--set-baseline", action="store_true",
                   help="record this report as the reference for the config version")
    g = sub.add_parser("gate"); g.add_argument("base"); g.add_argument("cand")
    g.add_argument("--identical", action="store_true", help="also require identical decisions")
    dg = sub.add_parser("diagnose"); dg.add_argument("--report")
    e = sub.add_parser("experiment"); e.add_argument("id")
    e.add_argument("--reassess", action="store_true", help="recompute verdicts from stored reports")
    pr = sub.add_parser("promote"); pr.add_argument("id"); pr.add_argument("--approved-by", required=True)
    rj = sub.add_parser("reject"); rj.add_argument("id"); rj.add_argument("--by", required=True)
    rj.add_argument("--why", required=True)
    rb = sub.add_parser("rollback"); rb.add_argument("version", nargs="?")
    sub.add_parser("state"); sub.add_parser("ingest")
    a = ap.parse_args(argv)

    from .util import BENCH_DIR, RUNS_DIR, read_json, rel
    if a.cmd == "datasets":
        from . import datasets
        if a.action == "build":
            for x in datasets.build_all():
                print(rel(x))
        else:
            print(rel(datasets.write_label_template()))
    elif a.cmd == "params":
        from . import params
        if a.action == "extract":
            print(rel(params.write_params()))
        else:
            drift = params.check_drift()
            print("\n".join(drift) or "params.yaml matches the code")
            return 1 if drift else 0
    elif a.cmd == "bench":
        from . import bench
        rc = _cfg(a.config)
        rep = bench.run(a.datasets.split(","), rc, name=a.name or rc.name)
        bench.print_summary(rep)
        if a.set_baseline:
            path = BENCH_DIR / "baselines.yaml"
            bl = yaml.safe_load(path.read_text()) if path.exists() else {}
            bl[rc.name] = rep["run_id"]
            path.write_text(yaml.safe_dump(bl, sort_keys=False))
    elif a.cmd == "gate":
        from . import bench
        ld = lambda r: read_json(Path(r) if r.endswith(".json") else BENCH_DIR / "reports" / f"{r}.json")
        res = bench.compare(ld(a.cand), ld(a.base))
        print(json.dumps({k: res[k] for k in ("gates", "new_wrong", "lost_correct", "new_correct",
                                              "latency")}, indent=1))
        need = ["no_new_wrong", "no_lost_correct", "latency_within_budget"] + (
            ["identical_decisions"] if a.identical else [])
        return 0 if all(res["gates"][k] for k in need) else 1
    elif a.cmd == "diagnose":
        from . import diagnose, memory
        rid = a.report or memory.baselines().get(_cfg(None).name)
        print(rel(diagnose.write_register(BENCH_DIR / "reports" / f"{rid}.json",
                                          RUNS_DIR / "traces" / f"{rid}.jsonl")))
    elif a.cmd == "experiment":
        from . import adapt, experiments
        res = experiments.reassess(a.id) if a.reassess else experiments.run(a.id)
        for v in res["variants"]:
            print(v["name"], "passed" if v["passed_gates"] else "FAILED", "improved" if v["improved"] else "")
        for pth in adapt.propose_from_experiment(res):
            print("proposal:", rel(pth))
    elif a.cmd == "promote":
        from . import validate
        print("active:", validate.promote(a.id, a.approved_by))
    elif a.cmd == "reject":
        from . import validate
        validate.reject(a.id, a.by, a.why)
    elif a.cmd == "rollback":
        from . import validate
        print("active:", validate.rollback(a.version))
    elif a.cmd == "state":
        from . import memory
        print(rel(memory.write_state()))
    elif a.cmd == "ingest":
        from . import monitor
        tr = monitor.ingest_captures()
        out = RUNS_DIR / "traces" / "captures.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(t) + "\n" for t in tr))
        print(f"{len(tr)} capture traces -> {rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
