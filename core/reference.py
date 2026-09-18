"""
Structured environmental data layer.

Turns a raw number or word into a *classified* observation, so the reasoning
engine works on interpreted classes ("critically_low SOC in a semi_arid zone")
rather than on bare numbers. This is the difference between a system that reads
data and one that understands it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.schema import Metric, Observation

REFERENCE_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "reference_data" / "thresholds.json"


@lru_cache(maxsize=1)
def load_reference() -> Dict[str, Any]:
    with open(REFERENCE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _band_for_value(metric_key: str, value: float) -> Optional[Dict[str, Any]]:
    ref = load_reference().get(metric_key, {})
    for band in ref.get("bands", []):
        if band["min"] <= value < band["max"]:
            return band
    return None


def _qualitative_lookup(metric_key: str, text: str) -> Optional[str]:
    ref = load_reference().get(metric_key, {})
    qmap = ref.get("qualitative_map", {})
    t = str(text).strip().lower()
    if t in qmap:
        return qmap[t]
    # longest-substring match so "low and erratic rainfall" still resolves
    best = None
    for key, cls in qmap.items():
        if key in t and (best is None or len(key) > len(best[0])):
            best = (key, cls)
    return best[1] if best else None


def classify(obs: Observation) -> Observation:
    """Attach a `classification` to an observation, in place."""
    key = obs.metric.value

    if obs.value is not None:
        band = _band_for_value(key, obs.value)
        if band:
            obs.classification = band["class"]
            return obs

    probe = obs.category or obs.raw
    if probe is not None:
        cls = _qualitative_lookup(key, str(probe))
        if cls:
            obs.classification = cls
            if obs.metric == Metric.LAND_USE:
                obs.category = cls
    return obs


def interpretation_for(metric: Metric, classification: Optional[str]) -> Optional[str]:
    """Human-readable meaning of a classification, straight from the reference table."""
    if classification is None:
        return None
    ref = load_reference().get(metric.value, {})
    for band in ref.get("bands", []):
        if band["class"] == classification:
            return band["interpretation"]
    if "class_interpretation" in ref:
        return ref["class_interpretation"].get(classification)
    classes = ref.get("classes", {})
    if classification in classes:
        return classes[classification].get("interpretation")
    return None


def land_use_attributes(land_use_class: Optional[str]) -> Dict[str, Any]:
    if not land_use_class:
        return {}
    return load_reference().get("land_use", {}).get("classes", {}).get(land_use_class, {})


def reference_citation(metric: Metric) -> Optional[str]:
    ref = load_reference().get(metric.value, {})
    return ref.get("reference")


def reference_source_id(metric: Metric) -> Optional[str]:
    ref = load_reference().get(metric.value, {})
    return ref.get("source_id")


# Ordered severity scales, used by the reasoning engine for comparisons.
SEVERITY_ORDER: Dict[str, List[str]] = {
    "soil_organic_carbon": ["critically_low", "very_low", "low", "moderate", "good", "very_high"],
    "soil_moisture": ["very_dry", "dry", "adequate", "wet", "saturated"],
    "rainfall": ["hyper_arid", "arid", "semi_arid", "dry_sub_humid", "humid", "very_humid"],
    "species_richness": ["very_low", "low", "moderate", "high", "very_high"],
    "habitat_diversity": ["very_low", "low", "moderate", "high"],
    "pollution": ["none", "low", "moderate", "high", "severe"],
    "deforestation": ["none", "low", "moderate", "high", "severe"],
    "temperature": ["cold", "cool_temperate", "warm_temperate", "warm", "hot"],
    "soil_ph": [
        "ultra_acidic", "extremely_acidic", "strongly_acidic", "moderately_acidic",
        "near_neutral", "moderately_alkaline", "strongly_alkaline",
    ],
}


def rank(metric: Metric, classification: Optional[str]) -> Optional[int]:
    """Position of a classification on its severity scale (low index = low end)."""
    if classification is None:
        return None
    scale = SEVERITY_ORDER.get(metric.value)
    if not scale or classification not in scale:
        return None
    return scale.index(classification)


def at_or_below(metric: Metric, classification: Optional[str], threshold: str) -> bool:
    r, t = rank(metric, classification), rank(metric, threshold)
    return r is not None and t is not None and r <= t


def at_or_above(metric: Metric, classification: Optional[str], threshold: str) -> bool:
    r, t = rank(metric, classification), rank(metric, threshold)
    return r is not None and t is not None and r >= t
