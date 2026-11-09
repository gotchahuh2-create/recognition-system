"""Parameter registry: every tuning constant in ``vision/``, versioned, overridable.

``vision/`` keeps its constants where they are (module-level or class-level
UPPER_CASE names). This module

* finds them by reading the source (``extract``), never by importing changes;
* writes them to ``config/params.yaml`` (version 1 = the code's values exactly);
* applies a set of overrides for one run (``apply``) by setting the attribute on
  the module or class and restoring it afterwards.

Binding. An override only works where the constant is read at call time. A
constant captured as a default argument (``def __init__(self, accept=ACCEPT)``)
or copied by ``from .x import NAME`` is *bound*: changing the module attribute
does not reach that use. ``extract`` marks those uses, and ``apply`` refuses to
pretend: it reports bound parameters instead of silently doing nothing. Making a
bound parameter tunable is a code change and needs an ADR.
"""

from __future__ import annotations

import ast
import importlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .util import CONFIG_DIR, ROOT, stable_hash

VISION = ROOT / "vision"
# Superseded code: not part of the parameter surface (docs/adr/0101).
LEGACY = ("vision/camera/", "vision/app.py")
PARAMS_FILE = CONFIG_DIR / "params.yaml"

# Bound constants that are still reachable after construction, because the
# recogniser copies them onto itself (self.accept = accept). An override of one of
# these is set on the built Pipeline instead of on the module (``apply_instance``).
INSTANCE = {
    "vision.board_ocr.ACCEPT": "boards.accept",
    "vision.board_ocr.MARGIN": "boards.margin",
    "vision.mcu_ocr.ACCEPT": "mcus.accept",
    "vision.mcu_ocr.MARGIN": "mcus.margin",
    "vision.mcu_ocr.CARD_ACCEPT": "mcus.card_accept",
    "vision.card_locate.ACCEPT": "locator.accept",
}


@dataclass
class Param:
    name: str            # dotted: "vision.mcu_ocr.ACCEPT" or "vision.pipeline.Pipeline.CARD_SURE"
    value: object
    module: str          # "vision.mcu_ocr"
    owner: str | None    # class name, or None for module level
    attr: str            # "ACCEPT"
    line: int
    comment: str = ""
    bound: list[str] = field(default_factory=list)  # where the value is captured, not read

    @property
    def live(self) -> bool:
        return not self.bound

    def as_yaml(self) -> dict:
        d = {"value": list(self.value) if isinstance(self.value, tuple) else self.value,
             "where": f"{self.module.replace('.', '/')}.py:{self.line}"}
        if self.comment:
            d["note"] = self.comment
        if self.bound:
            d["bound"] = self.bound
        if self.name in INSTANCE:
            d["via"] = "pipeline." + INSTANCE[self.name]
        return d


def _literal(node):
    try:
        v = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None
    if isinstance(v, bool) or isinstance(v, (int, float)):
        return v
    if isinstance(v, tuple) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                          for x in v):
        return v
    return None


def _is_const_name(n: str) -> bool:
    return n.isupper() and len(n) >= 2 and not n.startswith("_")


def _trailing_comment(lines: list[str], lineno: int) -> str:
    line = lines[lineno - 1]
    return line.split("#", 1)[1].strip() if "#" in line else ""


def _py_files():
    for p in sorted(VISION.rglob("*.py")):
        r = p.relative_to(ROOT).as_posix()
        if any(r.startswith(x) for x in LEGACY):
            continue
        yield p, r


def extract() -> list[Param]:
    """Read every tunable numeric constant in ``vision/`` from the source."""
    params: list[Param] = []
    trees = {}
    for path, r in _py_files():
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        trees[r] = tree
        lines = src.splitlines()
        mod = r[:-3].replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]

        def take(body, owner):
            for node in body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                        and isinstance(node.targets[0], ast.Name):
                    name, val = node.targets[0].id, _literal(node.value)
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                        and node.value is not None:
                    name, val = node.target.id, _literal(node.value)
                else:
                    continue
                if val is None or not _is_const_name(name):
                    continue
                dotted = f"{mod}.{owner}.{name}" if owner else f"{mod}.{name}"
                params.append(Param(dotted, val, mod, owner, name, node.lineno,
                                    _trailing_comment(lines, node.lineno)))

        take(tree.body, None)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                take(node.body, node.name)

    by_mod_attr = {(p.module, p.attr): p for p in params if p.owner is None}
    for r, tree in trees.items():
        mod = r[:-3].replace("/", ".").removesuffix(".__init__")
        for node in ast.walk(tree):
            # default arguments that capture a module constant at import time
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in node.args.defaults + [k for k in node.args.kw_defaults if k]:
                    for n in ast.walk(d):
                        if isinstance(n, ast.Name) and (mod, n.id) in by_mod_attr:
                            by_mod_attr[(mod, n.id)].bound.append(
                                f"default arg of {node.name}() in {r}:{node.lineno}")
            # copies made by `from .x import NAME`
            if isinstance(node, ast.ImportFrom) and node.module:
                base = mod.rsplit(".", node.level)[0] if node.level else ""
                src_mod = f"{base}.{node.module}" if node.level else node.module
                for a in node.names:
                    p = by_mod_attr.get((src_mod, a.name))
                    if p is not None:
                        p.bound.append(f"imported by value in {r}:{node.lineno}")
    return params


def to_yaml(params: list[Param], version: int = 1, note: str = "") -> dict:
    return {"version": version,
            "note": note or "Values exactly as in the code. Edit through a proposal "
                            "(phoenix.adapt), not by hand.",
            "fingerprint": stable_hash({p.name: p.value for p in params}),
            "params": {p.name: p.as_yaml() for p in params}}


def write_params(path: Path = PARAMS_FILE) -> Path:
    params = extract()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# PHOENIX parameter registry. Generated by `python -m phoenix params extract`.\n"
              "# 'bound' = the value is captured at import time; an override does NOT reach it.\n")
    path.write_text(header + yaml.safe_dump(to_yaml(params), sort_keys=False, width=100),
                    encoding="utf-8")
    return path


def load_params(path: Path = PARAMS_FILE) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _target(name: str):
    """Resolve a dotted parameter name to (object holding it, attribute)."""
    parts = name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        for p in parts[i:-1]:
            obj = getattr(obj, p)
        return obj, parts[-1]
    raise KeyError(name)


class BoundParameterError(ValueError):
    pass


@contextmanager
def apply(overrides: dict | None, registry: dict | None = None, strict: bool = True):
    """Temporarily set parameter values; restore them on exit.

    overrides: {"vision.mcu_ocr.ACCEPT": 0.88, ...}
    registry:  the loaded params.yaml, used to refuse unknown or bound names.
    strict:    raise on bound parameters (default); False = skip them with a note.
    Yields the list of (name, old, new) actually applied.
    """
    overrides = dict(overrides or {})
    known = (registry or load_params()).get("params", {})
    applied, skipped = [], []
    for name, new in overrides.items():
        if name not in known:
            raise KeyError(f"unknown parameter {name!r} (not in config/params.yaml)")
        if name in INSTANCE:
            continue  # set on the pipeline instance by apply_instance()
        if known[name].get("bound"):
            msg = f"{name} is bound ({'; '.join(known[name]['bound'])}): an override cannot reach it"
            if strict:
                raise BoundParameterError(msg)
            skipped.append(msg)
            continue
        obj, attr = _target(name)
        old = getattr(obj, attr)
        if isinstance(old, tuple) and isinstance(new, list):
            new = tuple(new)
        setattr(obj, attr, new)
        applied.append((name, old, new, obj, attr))
    try:
        yield [(n, o, v) for n, o, v, _, _ in applied] + [("skipped", s, None) for s in skipped]
    finally:
        for _, old, _, obj, attr in reversed(applied):
            setattr(obj, attr, old)


def check_drift(path: Path = PARAMS_FILE) -> list[str]:
    """Differences between config/params.yaml and the code (should be empty)."""
    reg = load_params(path)["params"]
    code = {p.name: p for p in extract()}
    out = [f"new in code: {n}" for n in code if n not in reg]
    out += [f"gone from code: {n}" for n in reg if n not in code]
    for n, p in code.items():
        if n in reg:
            v = list(p.value) if isinstance(p.value, tuple) else p.value
            if reg[n]["value"] != v:
                out.append(f"value differs: {n} code={v} registry={reg[n]['value']}")
    return out


def apply_instance(pipeline, overrides: dict | None) -> list[tuple]:
    """Set overrides of INSTANCE parameters on a built ``vision.pipeline.Pipeline``."""
    done = []
    for name, new in (overrides or {}).items():
        path = INSTANCE.get(name)
        if path is None:
            continue
        *owner, attr = path.split(".")
        obj = pipeline
        for o in owner:
            obj = getattr(obj, o)
        if obj is None:  # e.g. mcus is None when the MCU step is off
            continue
        done.append((name, getattr(obj, attr), new))
        setattr(obj, attr, new)
    return done
