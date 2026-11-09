"""Layer 1: build the recognition pipeline from a configuration, not from code edits.

A configuration is a small dict (usually ``config/active.yaml`` or an
experiment variant)::

    {"name": "baseline",
     "stages": {"chip_read": "classic", "card_localise": "frame_first", ...},
     "params": {"vision.mcu_ocr.ACCEPT": 0.90, ...}}   # overrides only

Stage choices must be implementations listed in ``config/registry.yaml``.
Parameter overrides must be names in ``config/params.yaml``. ``vision/``
itself is called, never modified.

Stage sequence (the order the brief requires; ``vision.pipeline.Pipeline.run``
already runs 1-5 in this order, the socket stage is added after it):

    1 preprocess  2 board  3 geometry (layout placement)  4 localise (MCU card,
    sockets)  5 recognise (card / MCU)  6 sockets (occupancy, Clicks)  7 manifest
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import params as P
from .util import CONFIG_DIR, stable_hash

REGISTRY_FILE = CONFIG_DIR / "registry.yaml"
ACTIVE_FILE = CONFIG_DIR / "active.yaml"


def load_registry(path: Path = REGISTRY_FILE) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def default_stages(registry: dict | None = None) -> dict:
    reg = registry or load_registry()
    return {k: str(v["default"]) for k, v in reg["stages"].items()}


@dataclass
class RunConfig:
    name: str
    stages: dict
    params: dict = field(default_factory=dict)
    note: str = ""

    @property
    def hash(self) -> str:
        return stable_hash({"stages": self.stages, "params": self.params})

    def as_dict(self) -> dict:
        return {"name": self.name, "hash": self.hash, "stages": self.stages,
                "params": self.params, **({"note": self.note} if self.note else {})}


def resolve(cfg: dict | None = None, registry: dict | None = None) -> RunConfig:
    """Fill a partial config with registry defaults and validate every choice."""
    reg = registry or load_registry()
    cfg = dict(cfg or {})
    stages = default_stages(reg)
    for k, v in (cfg.get("stages") or {}).items():
        if k not in reg["stages"]:
            raise KeyError(f"unknown stage {k!r}; known: {sorted(reg['stages'])}")
        v = str(v)
        if v not in {str(i) for i in reg["stages"][k]["impls"]}:
            raise KeyError(f"stage {k}: unknown implementation {v!r}; "
                           f"known: {sorted(map(str, reg['stages'][k]['impls']))}")
        stages[k] = v
    return RunConfig(cfg.get("name", "unnamed"), stages, dict(cfg.get("params") or {}),
                     cfg.get("note", ""))


def load_active() -> RunConfig:
    """The promoted configuration (config/active.yaml -> config/versions/vN.yaml)."""
    if not ACTIVE_FILE.exists():
        return resolve({"name": "defaults"})
    act = yaml.safe_load(ACTIVE_FILE.read_text(encoding="utf-8"))
    ver = yaml.safe_load((CONFIG_DIR / "versions" / f"{act['active']}.yaml")
                         .read_text(encoding="utf-8"))
    return resolve({"name": act["active"], **ver})


def experimental_choices(rc: RunConfig, registry: dict | None = None) -> list[str]:
    reg = registry or load_registry()
    out = []
    for k, v in rc.stages.items():
        impls = {str(i): d for i, d in reg["stages"][k]["impls"].items()}
        if impls[v].get("status") == "experimental":
            out.append(f"{k}={v}")
    return out


class PhoenixPipeline:
    """The protected ``vision`` pipeline + the configured stage choices + the socket stage."""

    def __init__(self, rc: RunConfig, catalog=None):
        from vision.pipeline import Pipeline
        self.rc = rc
        s = rc.stages
        self._params_ctx = None
        self.pipeline = Pipeline(
            catalog=catalog, backend=s["ocr_backend"],
            refine=s["refine"], layouts=_layouts_arg(s["layout"]),
            upscaler=s["upscaler"], chip_mode=s["chip_read"],
            verify_marking=(s["marking_verifier"] == "on"))
        if s["card_localise"] == "geometry_first":
            from .localise import GeometryFirstLocator
            self.pipeline.locator = GeometryFirstLocator(self.pipeline.locator.accept)
        self.instance_overrides = P.apply_instance(self.pipeline, rc.params)

    @contextmanager
    def active(self):
        """Module-level parameter overrides (and exact accelerations) apply inside this block."""
        from . import accel
        if self.rc.stages.get("text_match", "exact") == "exact_cached":
            accel.install()
        else:
            accel.uninstall()
        with P.apply(rc_params_module_level(self.rc.params)) as applied:
            yield applied

    def run(self, photo, **kw):
        with self.active():
            res = self.pipeline.run(photo, **kw)
            res.guards = []
            if self.rc.stages.get("consistency_guard", "off") == "on":
                from .guards import apply_all
                res.guards = apply_all(res, self.pipeline.catalog)
            if self.rc.stages["socket_occupancy"] != "off":
                from .sockets import sockets_from_result
                res.sockets = sockets_from_result(res, self.rc.stages["socket_occupancy"])
            else:
                res.sockets = []
        return res


def _layouts_arg(choice: str):
    from .util import ROOT
    if choice == "off":
        return False
    if choice == "on":
        return None  # vision default: data/layouts.json
    return ROOT / "data" / f"layouts_{choice}.json"


def rc_params_module_level(p: dict) -> dict:
    return {k: v for k, v in (p or {}).items() if k not in P.INSTANCE}
