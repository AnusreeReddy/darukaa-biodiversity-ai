"""
Unit tests.

    pytest -q

These test component behaviour. The end-to-end scenario assertions live in
eval/scenarios.yaml and run via `python -m eval.run_eval`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.conversation import Session
from core.evidence import attach_evidence
from core.extract import extract, extract_from_json
from core.reasoning import load_claims, reason
from core.reference import classify, interpretation_for
from core.render import to_json, to_markdown
from core.retrieve import kb_is_ready, search
from core.schema import Confidence, EnvState, Metric, Observation


# ---------------------------------------------------------------- reference

@pytest.mark.parametrize(
    "metric,value,expected",
    [
        (Metric.SOIL_ORGANIC_CARBON, 0.3, "critically_low"),
        (Metric.SOIL_ORGANIC_CARBON, 2.5, "moderate"),
        (Metric.SOIL_PH, 5.1, "strongly_acidic"),
        (Metric.SOIL_PH, 6.9, "near_neutral"),
        (Metric.SOIL_PH, 8.8, "strongly_alkaline"),
        (Metric.RAINFALL, 400, "semi_arid"),
        (Metric.RAINFALL, 1200, "humid"),
        (Metric.TEMPERATURE, 28, "hot"),
    ],
)
def test_numeric_classification(metric, value, expected):
    obs = classify(Observation(metric=metric, raw=value, value=float(value)))
    assert obs.classification == expected


def test_qualitative_classification():
    assert classify(Observation(metric=Metric.RAINFALL, raw="low and erratic")).classification == "semi_arid"
    assert classify(Observation(metric=Metric.LAND_USE, raw="monoculture wheat")).classification == "monoculture_cropland"


def test_every_classification_has_an_interpretation():
    for metric in [Metric.SOIL_ORGANIC_CARBON, Metric.SOIL_PH, Metric.RAINFALL]:
        obs = classify(Observation(metric=metric, raw=1.0, value=1.0))
        assert interpretation_for(metric, obs.classification)


# ---------------------------------------------------------------- extraction

def test_extract_free_text():
    obs = {o.metric: o for o in extract(
        "Soil organic carbon is 0.3%, rainfall is low, monoculture wheat, semi-arid, 28 C"
    )}
    assert obs[Metric.SOIL_ORGANIC_CARBON].value == 0.3
    assert obs[Metric.RAINFALL].classification == "semi_arid"
    assert obs[Metric.LAND_USE].category == "monoculture_cropland"
    assert obs[Metric.TEMPERATURE].value == 28.0


def test_organic_matter_converted_to_carbon():
    obs = {o.metric: o for o in extract("organic matter is 2.5%")}
    # 2.5 / 1.724 = 1.45
    assert obs[Metric.SOIL_ORGANIC_CARBON].value == pytest.approx(1.45, abs=0.01)


def test_extract_json():
    obs = {o.metric: o for o in extract_from_json(
        {"soil_ph": 6.4, "soc": "0.9%", "rainfall": 380, "land_use": "monoculture cotton", "pollution": "high"}
    )}
    assert obs[Metric.SOIL_PH].value == 6.4
    assert obs[Metric.SOIL_ORGANIC_CARBON].value == 0.9
    assert obs[Metric.RAINFALL].classification == "semi_arid"
    assert obs[Metric.POLLUTION].classification == "high"


def test_implausible_values_rejected():
    assert not any(o.metric == Metric.SOIL_PH for o in extract("pH 45"))


def test_deforestation_does_not_set_land_use_to_forest():
    obs = {o.metric: o for o in extract("We cleared the forest last year")}
    assert obs[Metric.DEFORESTATION].classification in {"high", "severe", "moderate"}
    assert Metric.LAND_USE not in obs


# ---------------------------------------------------------------- reasoning

def _state(**kwargs) -> EnvState:
    s = EnvState()
    for name, raw in kwargs.items():
        metric = Metric(name)
        value = float(raw) if isinstance(raw, (int, float)) else None
        s.set(classify(Observation(metric=metric, raw=raw, value=value)))
    return s


def test_reasoning_requires_three_variables():
    recs = reason(_state(soil_organic_carbon=0.3))
    assert recs == []


def test_acid_soil_gate_suppresses_legume_recommendation():
    s = _state(soil_ph=5.0, soil_organic_carbon=0.8, land_use="monoculture wheat", rainfall=900)
    ids = [r.intervention_id for r in reason(s)]
    assert "gate_acid_soil" in ids
    assert "action_legume_cover_crop" not in ids


def test_neutral_ph_restores_legume_recommendation():
    s = _state(soil_ph=6.9, soil_organic_carbon=0.8, land_use="monoculture wheat", rainfall=900)
    ids = [r.intervention_id for r in reason(s)]
    assert "action_legume_cover_crop" in ids
    assert "gate_acid_soil" not in ids


def test_dryland_gate_outranks_carbon_building():
    s = _state(rainfall=350, soil_organic_carbon=0.3, temperature=30, land_use="monoculture wheat")
    recs = reason(s)
    assert recs[0].intervention_id == "gate_dryland_water"


def test_every_recommendation_is_multi_metric():
    s = _state(soil_ph=6.5, soil_organic_carbon=0.6, rainfall=400, land_use="monoculture wheat",
               species_richness="low", pollution="high", temperature=27)
    for rec in reason(s):
        assert rec.n_metrics() >= 3, rec.intervention_id
        assert rec.interactions, rec.intervention_id


def test_recommendations_declare_horizon_and_confidence():
    s = _state(soil_organic_carbon=0.5, rainfall=500, land_use="monoculture wheat", species_richness="low")
    for rec in reason(s):
        assert rec.time_horizon.value in {"short", "medium", "long"}
        assert rec.confidence in {Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH}
        assert rec.confidence_reason


# ---------------------------------------------------------------- knowledge base

@pytest.mark.skipif(not kb_is_ready(), reason="knowledge base not built")
def test_retrieval_returns_attributed_passages():
    items = search("cover crops soil organic carbon sequestration rate", k=3)
    assert items
    for item in items:
        assert item.source_id and item.title and item.organisation and item.citation


@pytest.mark.skipif(not kb_is_ready(), reason="knowledge base not built")
def test_retrieval_can_be_restricted_to_sources():
    items = search("soil organic carbon", k=3, source_ids=["poeplau_don_2015"])
    assert items
    assert all(i.source_id == "poeplau_don_2015" for i in items)


def test_every_claim_declares_valid_sources():
    registry = yaml.safe_load((ROOT / "knowledge" / "sources.yaml").read_text(encoding="utf-8"))
    known = {s["id"] for s in registry["sources"]}
    for claim_id, claim in load_claims().items():
        assert claim["sources"], claim_id
        for src in claim["sources"]:
            assert src in known, f"claim '{claim_id}' cites unknown source '{src}'"


def test_every_source_has_a_corpus_file_and_citation():
    registry = yaml.safe_load((ROOT / "knowledge" / "sources.yaml").read_text(encoding="utf-8"))
    for source in registry["sources"]:
        assert (ROOT / "knowledge" / "corpus" / source["file"]).exists(), source["id"]
        assert source["citation"], source["id"]


@pytest.mark.skipif(not kb_is_ready(), reason="knowledge base not built")
def test_every_claim_retrieves_evidence():
    """Guards against a claim silently losing its supporting passage."""
    from core.retrieve import evidence_for_claim

    unsupported = [
        cid for cid, claim in load_claims().items()
        if not evidence_for_claim(claim["retrieval_query"], claim.get("sources", []), k=1)
    ]
    assert not unsupported, f"claims with no retrievable evidence: {unsupported}"


@pytest.mark.skipif(not kb_is_ready(), reason="knowledge base not built")
def test_unsupported_claim_is_dropped_and_confidence_falls():
    """A recommendation citing a non-existent claim must lose it, not assert it."""
    s = _state(soil_organic_carbon=0.5, rainfall=500, land_use="monoculture wheat", species_richness="low")
    recs = reason(s)
    target = recs[0]
    before = target.confidence
    target.claim_ids = target.claim_ids + ["this_claim_does_not_exist"]
    attach_evidence([target])
    assert "this_claim_does_not_exist" not in target.claim_ids
    assert target.confidence.value != before.value or before == Confidence.LOW


# ---------------------------------------------------------------- conversation

def test_single_variable_input_triggers_questions_not_advice():
    session = Session()
    resp = session.process("Biodiversity is declining on my land")
    assert resp.mode == "clarify"
    assert resp.questions
    assert not resp.recommendations
    assert all(q.why_it_matters for q in resp.questions)


def test_memory_accumulates_across_turns():
    session = Session()
    session.process("Biodiversity is declining on my land")
    session.process("Soil organic carbon is 0.4%")
    resp = session.process("Monoculture wheat, rainfall about 380mm")
    assert resp.mode == "recommend"
    assert session.state.value(Metric.SOIL_ORGANIC_CARBON) == 0.4
    assert session.state.classification(Metric.RAINFALL) == "semi_arid"
    assert len(session.state.known_core_metrics()) >= 4


def test_correction_overwrites_earlier_value():
    session = Session()
    session.process("pH 5.0, SOC 0.7%, monoculture wheat, rainfall 850mm, biodiversity low")
    resp = session.process("Actually the pH is 6.9")
    assert session.state.value(Metric.SOIL_PH) == 6.9
    ids = [r.intervention_id for r in resp.recommendations]
    assert "gate_acid_soil" not in ids


def test_session_reset_clears_state():
    session = Session()
    session.process("SOC 0.4%, monoculture wheat, rainfall 380mm")
    session.process("reset")
    assert session.state.observations == {}


def test_terse_numeric_answer_attributed_to_pending_question():
    session = Session()
    resp = session.process("Biodiversity is declining on my land")
    target = resp.questions[0].metric
    session.process("0.4" if target == Metric.SOIL_ORGANIC_CARBON else "3")
    assert session.state.has(target)


def test_implausible_terse_answer_is_not_recorded():
    session = Session()
    session.process("Biodiversity is declining on my land")
    before = dict(session.state.observations)
    session.process("999999")
    assert set(session.state.observations) == set(before)


# ---------------------------------------------------------------- output contract

@pytest.mark.skipif(not kb_is_ready(), reason="knowledge base not built")
def test_output_contains_every_required_field():
    session = Session()
    resp = session.process("SOC 0.3%, rainfall low, monoculture wheat, semi-arid, 28 C")
    payload = to_json(resp)
    assert payload["recommendations"]
    for rec in payload["recommendations"]:
        assert rec["action"] and rec["mechanism"]
        assert rec["impacted_metrics"]
        assert rec["time_horizon"] in {"short", "medium", "long"}
        assert rec["confidence"] in {"low", "medium", "high"}
        assert rec["variable_count"] >= 3
        assert rec["cross_variable_interactions"]
        assert rec["evidence"], rec["id"]
        for item in rec["evidence"]:
            assert item["source_id"] and item["citation"] and item["passage"]


@pytest.mark.skipif(not kb_is_ready(), reason="knowledge base not built")
def test_markdown_render_includes_sources_section():
    session = Session()
    resp = session.process("SOC 0.3%, rainfall low, monoculture wheat, semi-arid, 28 C")
    md = to_markdown(resp)
    for heading in ["What to do", "Why it works", "Impacted metrics", "Time horizon",
                    "Confidence", "Evidence", "Sources cited in this response"]:
        assert heading in md
