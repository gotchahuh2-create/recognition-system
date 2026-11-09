"""Validation and Recovery: gates, versioned configuration, promotion, rollback, safe runs.

Config versions: ``config/versions/vN.yaml`` (stages + param overrides, never
edited after creation) and ``config/active.yaml``::

    active: v2
    previous: v1
    last_known_good: v1     # the version recovery falls back to
    history: [{version, date, by, proposal}]

Promotion (``promote``) requires a proposal that passed its gates and a named
approver; it writes v(N+1), switches ``active``, records an ADR stub. Rollback
switches ``active`` back to any earlier version: no code changes, no deletes.

``SafeRunner`` is runtime recovery: if a run with the active config raises,
the photo is re-run with ``last_known_good`` and the failure is traced.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import yaml

from . import adapt
from .stages import ACTIVE_FILE, PhoenixPipeline, load_active, resolve
from .util import CONFIG_DIR, DOCS_DIR

VERSIONS = CONFIG_DIR / "versions"
REQUIRED_GATES = ("no_new_wrong", "no_lost_correct", "latency_within_budget")


def _read_active() -> dict:
    return yaml.safe_load(ACTIVE_FILE.read_text(encoding="utf-8"))


def _write_active(a: dict) -> None:
    ACTIVE_FILE.write_text(
        "# Which config version recognition uses. Change only with `python -m phoenix "
        "promote|rollback`.\n" + yaml.safe_dump(a, sort_keys=False), encoding="utf-8")


def init_versions() -> None:
    """v1 = registry defaults and no overrides = exactly the v0.10 behaviour."""
    VERSIONS.mkdir(parents=True, exist_ok=True)
    v1 = VERSIONS / "v1.yaml"
    if not v1.exists():
        rc = resolve({"name": "v1"})
        v1.write_text("# v0.10 behaviour: registry defaults, no parameter overrides.\n"
                      + yaml.safe_dump({"stages": rc.stages, "params": {}}, sort_keys=False),
                      encoding="utf-8")
    if not ACTIVE_FILE.exists():
        _write_active({"active": "v1", "previous": None, "last_known_good": "v1",
                       "history": [{"version": "v1", "date": time.strftime("%Y-%m-%d"),
                                    "by": "baseline", "proposal": None}]})


def gate_ok(gates: dict) -> bool:
    return all(gates.get(k) for k in REQUIRED_GATES)


def promote(pid: str, approved_by: str) -> str:
    if not approved_by.strip():
        raise ValueError("promotion needs the approver's name")
    p = adapt.load(pid)
    if p["status"] not in ("pending", "approved"):
        raise ValueError(f"{pid} is {p['status']}")
    if not gate_ok(p["evidence"]["gates"]):
        raise ValueError(f"{pid} did not pass its gates: {p['evidence']['gates']}")
    act = _read_active()
    if p["from_version"] != act["active"]:
        raise ValueError(f"{pid} was measured against {p['from_version']}, active is "
                         f"{act['active']}: re-run the experiment first")
    cur = load_active()
    n = max(int(f.stem[1:]) for f in VERSIONS.glob("v*.yaml")) + 1
    new = f"v{n}"
    body = {"stages": {**cur.stages, **p["change"]["stages"]},
            "params": {**cur.params, **p["change"]["params"]}}
    (VERSIONS / f"{new}.yaml").write_text(
        f"# Promoted from {pid} by {approved_by} on {time.strftime('%Y-%m-%d')}.\n"
        + yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    act["history"].append({"version": new, "date": time.strftime("%Y-%m-%d"),
                           "by": approved_by, "proposal": pid})
    act.update({"previous": act["active"], "last_known_good": act["active"], "active": new})
    _write_active(act)
    p.update({"status": "promoted", "approved_by": approved_by,
              "decided": time.strftime("%Y-%m-%d %H:%M"), "version": new})
    adapt.save(p)
    _adr_stub(p, new)
    return new


def reject(pid: str, by: str, why: str) -> None:
    p = adapt.load(pid)
    p.update({"status": "rejected", "approved_by": None, "rejected_by": by, "why": why,
              "decided": time.strftime("%Y-%m-%d %H:%M")})
    adapt.save(p)


def rollback(version: str | None = None) -> str:
    act = _read_active()
    target = version or act.get("previous") or act["last_known_good"]
    if not (VERSIONS / f"{target}.yaml").exists():
        raise FileNotFoundError(target)
    act["history"].append({"version": target, "date": time.strftime("%Y-%m-%d"),
                           "by": "rollback", "proposal": None, "from": act["active"]})
    act.update({"previous": act["active"], "active": target})
    _write_active(act)
    return target


def _adr_stub(p: dict, version: str) -> Path:
    adr = DOCS_DIR / "adr"
    nums = [int(f.name[:4]) for f in adr.glob("0*.md") if f.name[:4].isdigit()]
    n = max([x for x in nums if x >= 100] or [99]) + 1
    path = adr / f"{n:04d}-promote-{p['id'].lower()}.md"
    path.write_text(
        f"# ADR-{n:04d}: Promote {p['id']} as config {version}\n\n- Date: {time.strftime('%Y-%m-%d')}\n"
        f"- Status: Accepted\n- Approved by: {p['approved_by']}\n\n## Decision\n\n"
        f"Stages {p['change']['stages'] or '{}'}; params {p['change']['params'] or '{}'}.\n\n"
        f"## Evidence\n\nExperiment {p['experiment']} (hypothesis {p['hypothesis']}); gates "
        f"{p['evidence']['gates']}; new correct {len(p['evidence']['new_correct'])}.\n\n"
        f"## Rollback\n\n`python -m phoenix rollback {p['from_version']}`\n", encoding="utf-8")
    return path


class SafeRunner:
    """Run photos with the active config; on an exception fall back to last-known-good."""

    def __init__(self):
        act = _read_active() if ACTIVE_FILE.exists() else {"active": "v1", "last_known_good": "v1"}
        self.active = load_active()
        self.lkg_name = act["last_known_good"]
        self._pp = PhoenixPipeline(self.active)
        self._lkg = None
        self.failures: list[dict] = []

    def run(self, photo, **kw):
        try:
            return self._pp.run(photo, **kw), self.active.name
        except Exception as e:  # noqa: BLE001 - recovery must catch everything
            self.failures.append({"photo": getattr(photo, "name", "?"), "config": self.active.name,
                                  "error": repr(e), "trace": traceback.format_exc(limit=5)})
            if self.lkg_name == self.active.name:
                raise
            if self._lkg is None:
                ver = yaml.safe_load((VERSIONS / f"{self.lkg_name}.yaml").read_text(encoding="utf-8"))
                self._lkg = PhoenixPipeline(resolve({"name": self.lkg_name, **ver}))
            return self._lkg.run(photo, **kw), self.lkg_name
