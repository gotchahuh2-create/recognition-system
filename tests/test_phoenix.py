"""Tests for the PHOENIX engineering layer (no OCR runs; fast)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from phoenix import bench, diagnose, params, sockets, stages


# ---------------------------------------------------------------- params
def test_registry_matches_code():
    assert params.check_drift() == []


def test_extract_finds_decision_thresholds_and_marks_bound():
    ps = {p.name: p for p in params.extract()}
    assert ps["vision.mcu_ocr.ACCEPT"].value == 0.90
    assert ps["vision.mcu_ocr.ACCEPT"].bound  # default arg of McuRecognizer.__init__
    assert ps["vision.pipeline.Pipeline.CARD_SURE"].live
    assert not any(n.startswith("vision.camera") for n in ps)  # legacy excluded (ADR-0101)


def test_apply_overrides_and_restores():
    from vision.pipeline import Pipeline
    old = Pipeline.CARD_SURE
    with params.apply({"vision.pipeline.Pipeline.CARD_SURE": 0.5}) as applied:
        assert Pipeline.CARD_SURE == 0.5
        assert applied[0][0] == "vision.pipeline.Pipeline.CARD_SURE"
    assert Pipeline.CARD_SURE == old


def test_apply_restores_after_exception():
    from vision import card_locate
    old = card_locate.WORK_SIDE
    with pytest.raises(RuntimeError):
        with params.apply({"vision.card_locate.WORK_SIDE": 10}):
            raise RuntimeError
    assert card_locate.WORK_SIDE == old


def test_bound_parameter_is_refused_not_silently_ignored():
    with pytest.raises(params.BoundParameterError):
        with params.apply({"vision.enhance.TARGET_GLYPH_PX": 30}):
            pass


def test_unknown_parameter_is_refused():
    with pytest.raises(KeyError):
        with params.apply({"vision.nope.X": 1}):
            pass


def test_instance_parameters_reach_the_built_recognisers():
    class Obj:
        pass
    pl = Obj(); pl.mcus = Obj(); pl.mcus.accept = 0.9; pl.boards = Obj(); pl.boards.accept = 0.72
    done = params.apply_instance(pl, {"vision.mcu_ocr.ACCEPT": 0.95, "vision.board_ocr.ACCEPT": 0.8})
    assert pl.mcus.accept == 0.95 and pl.boards.accept == 0.8 and len(done) == 2


# ---------------------------------------------------------------- stages / registry
def test_registry_protects_the_six_components():
    reg = stages.load_registry()
    comps = {d.get("component") for s in reg["stages"].values() for d in s["impls"].values()
             if d.get("status") == "protected"}
    for c in ("Standard Board OCR", "MCU Reader", "Resolution Booster",
              "MCU and Socket Identifiers", "Guided Scanning"):
        assert c in comps
    faint = reg["stages"]["chip_read"]["impls"]["faint"]
    assert faint["component"] == "Pixel-by-pixel clustering"


def test_defaults_are_the_v010_behaviour():
    rc = stages.resolve({})
    assert rc.stages["chip_read"] == "classic" and rc.stages["card_localise"] == "frame_first"
    assert rc.stages["socket_occupancy"] == "off" and rc.params == {}


def test_resolve_rejects_unknown_choices():
    with pytest.raises(KeyError):
        stages.resolve({"stages": {"chip_read": "magic"}})
    with pytest.raises(KeyError):
        stages.resolve({"stages": {"nope": "x"}})


def test_config_hash_is_order_independent():
    a = stages.resolve({"params": {"vision.mcu_ocr.ACCEPT": 0.9, "vision.board_ocr.ACCEPT": 0.7}})
    b = stages.resolve({"params": {"vision.board_ocr.ACCEPT": 0.7, "vision.mcu_ocr.ACCEPT": 0.9}})
    assert a.hash == b.hash


# ---------------------------------------------------------------- bench scoring and gates
def _trace(**dec):
    return {"decisions_flat": {"board": None, "mcu": None, "mcu_card": None,
                               "mcu_card_layout": None, **dec},
            "top3": {"board": [], "mcu": ["STM32F429ZI"], "mcu_card": []}}


def test_outcomes_abstain_is_not_wrong_and_none_truth():
    assert bench.outcome("mcu", _trace(), "STM32F429ZI") == "abstain"
    assert bench.outcome("mcu", _trace(mcu="STM32F429ZI"), "stm32f429zi") == "correct"
    assert bench.outcome("mcu", _trace(mcu="STM32F439ZI"), "STM32F429ZI") == "wrong"
    assert bench.outcome("board", _trace(), "none") == "correct"
    assert bench.outcome("board", _trace(board="easypic-v8"), "none") == "wrong"


def test_unverified_items_measure_agreement_only():
    sc = bench.score_item(_trace(board="uni-ds-v8"), {}, {"board": "uni-ds-v8"})
    assert sc["board"]["outcome"] == "unverified" and sc["board"]["agree"]
    s = bench.summarise([{"item": "a", "scores": sc, "wall_ms": 10, "timings_ms": {}}])
    assert s["fields"]["board"]["n"] == 0 and s["fields"]["board"]["agree_provisional"] == 1


def _report(rid, dec, outcome, p95=100):
    return {"run_id": rid, "datasets": {"d": {"n": 1, "latency_ms": {"p95": p95, "mean": p95}}},
            "items": {"d/x": {"decisions": dec, "scores": {"mcu": {"outcome": outcome}}}}}


def test_compare_gates():
    base = _report("b", {"mcu": None}, "abstain")
    better = _report("c", {"mcu": "X"}, "correct")
    wrong = _report("w", {"mcu": "Y"}, "wrong")
    slow = _report("s", {"mcu": None}, "abstain", p95=200)
    g = bench.compare(better, base)
    assert g["gates"]["no_new_wrong"] and g["new_correct"] == ["d/x:mcu"]
    assert not g["gates"]["identical_decisions"]
    assert not bench.compare(wrong, base)["gates"]["no_new_wrong"]
    assert not bench.compare(base, better)["gates"]["no_lost_correct"]
    assert not bench.compare(slow, base)["gates"]["latency_within_budget"]
    assert bench.compare(base, base)["gates"]["identical_decisions"]


# ---------------------------------------------------------------- diagnose
def test_diagnose_names_wrong_answers_first():
    rep = {"run_id": "r", "items": {
        "d/a": {"scores": {"mcu_card_layout": {"outcome": "wrong"}}, "reasons": {}, "wall_ms": 1},
        "d/b": {"scores": {"mcu": {"outcome": "abstain"}},
                "reasons": {"mcu": "x; the MCU card is only 246 px in this photo"}, "wall_ms": 1}}}
    ws = diagnose.diagnose(rep)
    assert ws[0].severity == "critical"
    assert {w.id for w in ws} >= {"W-WRONG", "W-LAYOUT-GROUP", "W-SMALL-CARD"}


# ---------------------------------------------------------------- sockets
def _area(color, yellow_lines=True, size=(300, 120)):
    img = np.zeros((size[0], size[1], 3), np.uint8)
    img[:] = (20, 20, 20)  # dark solder mask
    if yellow_lines:
        img[::30, :] = (0, 210, 230)  # yellow silkscreen rows (BGR)
    if color is not None:
        img[30:270, 10:110] = color
    return img


def test_socket_occupancy_three_valued():
    occ, s, _, f = sockets.classify(_area((60, 160, 40)))      # green Click PCB
    assert occ == sockets.Occupancy.OCCUPIED and s > 0.5
    occ, s, _, _ = sockets.classify(_area(None))               # bare board
    assert occ == sockets.Occupancy.EMPTY
    occ, _, why, _ = sockets.classify(_area(None), visible=0.5)
    assert occ == sockets.Occupancy.UNKNOWN and "inside the photo" in why


def test_derive_sockets_csv(tmp_path):
    lay = tmp_path / "layouts.json"
    lay.write_text(json.dumps({"_doc": "x", "b": {"sockets": [{"slot": "mikroBUS 1",
                                                               "box": [0.1, 0.2, 0.3, 0.4]}]}}))
    n = sockets.derive_sockets_csv(lay, tmp_path / "s.csv")
    assert n == 1 and "b,mikroBUS 1,0.2,0.3" in (tmp_path / "s.csv").read_text()


# ---------------------------------------------------------------- validate (promotion / rollback)
def test_promote_and_rollback(tmp_path, monkeypatch):
    from phoenix import adapt, validate
    vers = tmp_path / "versions"; vers.mkdir()
    act = tmp_path / "active.yaml"
    monkeypatch.setattr(validate, "VERSIONS", vers)
    monkeypatch.setattr(validate, "ACTIVE_FILE", act)
    monkeypatch.setattr(stages, "ACTIVE_FILE", act)
    monkeypatch.setattr(stages, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(adapt, "PROP_DIR", tmp_path / "proposals")
    monkeypatch.setattr(validate, "DOCS_DIR", tmp_path)
    (tmp_path / "adr").mkdir()
    validate.init_versions()
    assert yaml.safe_load(act.read_text())["active"] == "v1"
    prop = {"id": "P-001", "status": "pending", "experiment": "E-1", "hypothesis": "H-1",
            "change": {"stages": {"socket_occupancy": "heuristic_v0"}, "params": {}},
            "from_version": "v1", "risks": [],
            "evidence": {"baseline_report": "b", "variant_report": "c",
                         "gates": {"no_new_wrong": True, "no_lost_correct": True,
                                   "latency_within_budget": True},
                         "new_correct": [], "lost_correct": [], "new_wrong": [], "latency": {}}}
    adapt.save(prop)
    with pytest.raises(ValueError):
        validate.promote("P-001", "  ")
    assert validate.promote("P-001", "Owner") == "v2"
    assert stages.load_active().stages["socket_occupancy"] == "heuristic_v0"
    assert adapt.load("P-001")["status"] == "promoted"
    assert validate.rollback() == "v1"
    assert stages.load_active().stages["socket_occupancy"] == "off"


def test_failed_gates_cannot_be_promoted(tmp_path, monkeypatch):
    from phoenix import adapt, validate
    monkeypatch.setattr(adapt, "PROP_DIR", tmp_path)
    adapt.save({"id": "P-009", "status": "pending", "experiment": "E", "hypothesis": "H",
                "change": {"stages": {}, "params": {}}, "from_version": "v1", "risks": [],
                "evidence": {"baseline_report": "b", "variant_report": "c",
                             "gates": {"no_new_wrong": False, "no_lost_correct": True,
                                       "latency_within_budget": True},
                             "new_correct": [], "lost_correct": [], "new_wrong": ["x"],
                             "latency": {}}})
    with pytest.raises(ValueError):
        validate.promote("P-009", "Owner")


def test_geometry_first_without_layout_equals_default():
    from vision.card_locate import CardLocator
    from phoenix.localise import GeometryFirstLocator
    img = np.zeros((400, 600, 3), np.uint8)
    img[100:300, 200:400] = (20, 20, 20)
    a = CardLocator().locate(img)
    b = GeometryFirstLocator().locate(img)
    assert [(x.method, round(x.score, 4)) for x in a] == [(x.method, round(x.score, 4)) for x in b]


# ---------------------------------------------------------------- guards (H-007)
def test_guard_withdraws_card_that_cannot_carry_the_mcu():
    from types import SimpleNamespace as NS
    from vision.catalog import Catalog
    from vision.results import Candidate, Recognition
    from phoenix.guards import card_carries_mcu
    cat = Catalog.load()
    empty = next(c for c in cat.mcu_cards if c.empty and c.mcus)
    full = next(c for c in cat.mcu_cards if c.mcu)
    mk = lambda kind, key: Recognition(kind, [Candidate(key, key, 0.95)], Candidate(key, key, 0.95), "r")
    res = NS(mcu=mk("mcu", "STM32F303RC"), mcu_card=mk("mcu_card", empty.key))
    assert card_carries_mcu(res, cat) and res.mcu_card.decision is None
    res = NS(mcu=mk("mcu", full.mcu), mcu_card=mk("mcu_card", full.key))
    assert card_carries_mcu(res, cat) is None and res.mcu_card.decision is not None


def test_socket_v1_and_scoring():
    occ, _, _, _ = sockets.classify_v1(_area((60, 160, 40)))
    assert occ == sockets.Occupancy.OCCUPIED
    occ, _, _, _ = sockets.classify_v1(_area(None))
    assert occ == sockets.Occupancy.EMPTY
    tr = {"sockets": [{"slot": "mikroBUS 1", "occupancy": "occupied"},
                      {"slot": "mikroBUS 2", "occupancy": "unknown"},
                      {"slot": "mikroBUS 3", "occupancy": "empty"}]}
    assert bench.score_sockets(tr, {"mikroBUS 1": "occupied", "mikroBUS 2": "empty",
                                    "mikroBUS 3": "occupied"}) == \
        {"mikroBUS 1": "correct", "mikroBUS 2": "abstain", "mikroBUS 3": "wrong"}
