"""
Input layer: free text and structured JSON -> classified Observations.

Extraction is deterministic (patterns + a controlled vocabulary), not
model-based. That choice is deliberate: extraction errors here propagate into
every downstream recommendation, and a deterministic parser is testable,
reproducible and cannot invent a value that the user did not supply. An
optional LLM pass (core/llm.py) handles phrasing the parser misses, but it can
only ever *add* observations for metrics still missing, never overwrite a
parsed measurement.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from core.reference import classify
from core.schema import Metric, Observation

# --------------------------------------------------------------------------
# Numeric patterns
# --------------------------------------------------------------------------

NUM = r"(-?\d+(?:\.\d+)?)"

NUMERIC_PATTERNS: List[Tuple[Metric, str, Optional[str]]] = [
    # soil pH
    (Metric.SOIL_PH, rf"\bp\.?h\b[^0-9\-]{{0,18}}{NUM}", "pH"),
    (Metric.SOIL_PH, rf"{NUM}\s*p\.?h\b", "pH"),
    # soil organic carbon
    (Metric.SOIL_ORGANIC_CARBON, rf"\b(?:soil\s+)?organic\s+carbon\b[^0-9\-]{{0,20}}{NUM}\s*%", "%"),
    (Metric.SOIL_ORGANIC_CARBON, rf"\bsoc\b[^0-9\-]{{0,20}}{NUM}\s*%?", "%"),
    (Metric.SOIL_ORGANIC_CARBON, rf"\borganic\s+matter\b[^0-9\-]{{0,20}}{NUM}\s*%", "%_om"),
    (Metric.SOIL_ORGANIC_CARBON, rf"{NUM}\s*%\s*(?:soil\s+)?organic\s+carbon", "%"),
    # rainfall
    (Metric.RAINFALL, rf"\brain(?:fall)?\b[^0-9\-]{{0,25}}{NUM}\s*mm", "mm/yr"),
    (Metric.RAINFALL, rf"\bprecipitation\b[^0-9\-]{{0,25}}{NUM}\s*mm", "mm/yr"),
    (Metric.RAINFALL, rf"{NUM}\s*mm\s*(?:of\s*)?(?:annual\s*)?rain", "mm/yr"),
    (Metric.RAINFALL, rf"{NUM}\s*mm\s*(?:per\s*(?:year|annum)|/\s*(?:yr|year))", "mm/yr"),
    # temperature
    (Metric.TEMPERATURE, rf"\btemp(?:erature)?\b[^0-9\-]{{0,25}}{NUM}\s*(?:°|deg(?:rees)?\s*)?c\b", "degC"),
    (Metric.TEMPERATURE, rf"{NUM}\s*(?:°\s*c|deg(?:rees)?\s*c)\b", "degC"),
    (Metric.TEMPERATURE, rf"{NUM}\s*°?\s*c\b(?!\w)", "degC"),
    # soil moisture
    (Metric.SOIL_MOISTURE, rf"\b(?:soil\s+)?moisture\b[^0-9\-]{{0,20}}{NUM}\s*%", "% VWC"),
    (Metric.SOIL_MOISTURE, rf"\bvwc\b[^0-9\-]{{0,15}}{NUM}\s*%?", "% VWC"),
    # species richness
    (Metric.SPECIES_RICHNESS, rf"{NUM}\s*(?:different\s+)?species\b", "species"),
    (Metric.SPECIES_RICHNESS, rf"\bspecies\s+(?:richness|count)\b[^0-9\-]{{0,18}}{NUM}", "species"),
    # habitat diversity
    (Metric.HABITAT_DIVERSITY, rf"{NUM}\s*(?:distinct\s+)?habitat(?:\s+types?)?\b", "habitat types"),
    # deforestation
    (Metric.DEFORESTATION, rf"\b(?:lost|cleared|removed)\b[^0-9\-]{{0,20}}{NUM}\s*%[^.]{{0,30}}(?:tree|forest|wood)", "%"),
    (Metric.DEFORESTATION, rf"{NUM}\s*%\s*(?:of\s*)?(?:tree|forest|woody)\s*(?:cover\s*)?loss", "%"),
    # farm size
    (Metric.FARM_SIZE, rf"{NUM}\s*(?:ha\b|hectares?)", "ha"),
    (Metric.FARM_SIZE, rf"{NUM}\s*acres?", "acre"),
]

# --------------------------------------------------------------------------
# Qualitative vocabulary: metric -> phrase -> normalised word
# --------------------------------------------------------------------------

QUALITATIVE_CUES: Dict[Metric, Dict[str, str]] = {
    Metric.RAINFALL: {
        "low rainfall": "low", "rainfall is low": "low", "little rain": "low",
        "scarce rain": "low", "erratic rain": "erratic", "irregular rain": "erratic",
        "poor rain": "low", "rainfall low": "low", "high rainfall": "high",
        "heavy rain": "heavy", "good rain": "good", "moderate rain": "moderate",
        "monsoon": "monsoon", "drought": "low", "dry season": "low",
    },
    Metric.SOIL_MOISTURE: {
        "soil is dry": "dry", "dry soil": "dry", "moisture is low": "dry",
        "low moisture": "dry", "waterlogged": "waterlogged", "saturated soil": "saturated",
        "soil is wet": "wet", "moist soil": "moist", "adequate moisture": "adequate",
        "good moisture": "adequate", "very dry": "very dry",
    },
    Metric.SPECIES_RICHNESS: {
        "biodiversity is declining": "declining", "biodiversity declining": "declining",
        "losing species": "declining", "fewer species": "low", "species declining": "declining",
        "low biodiversity": "low", "poor biodiversity": "low", "no birds": "low",
        "few pollinators": "low", "fewer pollinators": "low", "no pollinators": "very low",
        "rich biodiversity": "high", "high biodiversity": "high", "diverse wildlife": "high",
        "biodiversity is good": "high", "few species": "low",
        "species richness is low": "low", "richness is low": "low",
        "species richness low": "low", "richness is high": "high",
        "species richness is high": "high", "species richness is moderate": "moderate",
    },
    Metric.HABITAT_DIVERSITY: {
        "no hedgerows": "very low", "no trees": "very low", "no field margins": "very low",
        "uniform landscape": "uniform", "homogeneous": "homogeneous",
        "single habitat": "very low", "one habitat": "very low",
        "mixed habitat": "mixed", "varied habitat": "diverse", "habitat mosaic": "mosaic",
        "some hedgerows": "moderate", "wetland and woodland": "diverse",
        "habitat diversity is low": "low", "habitat diversity low": "low",
        "habitat diversity is very low": "very low", "habitat diversity is high": "high",
        "little non-crop habitat": "very low", "no non-crop habitat": "very low",
    },
    Metric.POLLUTION: {
        "spray insecticide": "high", "insecticide weekly": "high",
        "weekly spray": "high", "spraying weekly": "high",
        "insecticide": "moderate", "pesticide": "moderate", "fungicide": "moderate",
        "herbicide": "moderate", "agrochemical": "moderate",
        "heavy pesticide": "high", "heavy fertiliser": "high", "heavy fertilizer": "high",
        "high pesticide": "high", "lot of pesticide": "high", "lots of pesticide": "high",
        "heavy chemical": "high", "intensive spraying": "high", "spray frequently": "high",
        "spray weekly": "high", "moderate pesticide": "moderate", "some pesticide": "moderate",
        "low pesticide": "low", "minimal pesticide": "low", "no pesticide": "none",
        "no chemicals": "none", "organic farming": "none", "chemical free": "none",
        "industrial runoff": "severe", "contaminated": "severe", "heavy metal": "severe",
        "nutrient runoff": "moderate", "eutrophication": "high", "pesticide use is high": "high",
    },
    Metric.DEFORESTATION: {
        "cleared the forest": "high", "forest cleared": "high", "cut down trees": "high",
        "removed trees": "high", "removed the trees": "high", "tree cover loss": "moderate",
        "deforestation": "moderate", "heavy deforestation": "high", "severe deforestation": "severe",
        "no deforestation": "none", "forest intact": "none", "no tree loss": "none",
        "recent clearing": "high", "logged": "high",
    },
    Metric.LAND_USE: {
        "monoculture": "monoculture", "single crop": "monoculture", "mono crop": "monoculture",
        "crop rotation": "rotation", "rotating crops": "rotation",
        "mixed cropping": "mixed cropping", "intercropping": "intercropping",
        "agroforestry": "agroforestry", "silvopasture": "silvopasture",
        "orchard": "orchard", "plantation": "plantation",
        "grazing": "grazing", "pasture": "pasture", "rangeland": "rangeland",
        "livestock": "livestock", "degraded": "degraded", "barren": "barren",
        "wasteland": "wasteland", "fallow": "fallow", "wetland": "wetland",
        "forest": "forest", "paddy": "rice", "wheat": "wheat", "rice": "rice",
        "maize": "maize", "corn": "corn", "cotton": "cotton", "sugarcane": "sugarcane",
        "tea": "tea", "coffee": "coffee", "soybean": "soybean",
    },
    Metric.TEMPERATURE: {
        "very hot": "very hot", "hot climate": "hot", "hot summers": "hot",
        "warm climate": "warm", "cool climate": "cool", "cold climate": "cold",
        "mild climate": "mild", "temperate climate": "mild",
    },
    Metric.CLIMATE_ZONE: {
        "semi-arid": "semi-arid", "semi arid": "semi-arid", "semiarid": "semi-arid",
        "arid": "arid", "desert": "desert", "sub-humid": "sub-humid",
        "humid": "humid", "tropical": "tropical", "temperate": "temperate",
        "mediterranean": "mediterranean", "monsoon": "monsoon", "coastal": "coastal",
    },
    Metric.IRRIGATION: {
        "irrigated": "irrigated", "irrigation": "irrigated", "rainfed": "rainfed",
        "rain-fed": "rainfed", "no irrigation": "rainfed", "drip irrigation": "drip",
        "borewell": "groundwater", "tube well": "groundwater", "canal": "canal",
    },
}

# Keys accepted in structured JSON input.
JSON_KEY_MAP: Dict[str, Metric] = {
    "soil_ph": Metric.SOIL_PH, "ph": Metric.SOIL_PH, "soilph": Metric.SOIL_PH,
    "soil_organic_carbon": Metric.SOIL_ORGANIC_CARBON, "soc": Metric.SOIL_ORGANIC_CARBON,
    "organic_carbon": Metric.SOIL_ORGANIC_CARBON, "soil_carbon": Metric.SOIL_ORGANIC_CARBON,
    "soil_moisture": Metric.SOIL_MOISTURE, "moisture": Metric.SOIL_MOISTURE, "vwc": Metric.SOIL_MOISTURE,
    "land_use": Metric.LAND_USE, "landuse": Metric.LAND_USE, "land_cover": Metric.LAND_USE,
    "crop": Metric.LAND_USE, "cropping_system": Metric.LAND_USE, "land_use_type": Metric.LAND_USE,
    "species_richness": Metric.SPECIES_RICHNESS, "biodiversity": Metric.SPECIES_RICHNESS,
    "species_count": Metric.SPECIES_RICHNESS,
    "habitat_diversity": Metric.HABITAT_DIVERSITY, "habitat": Metric.HABITAT_DIVERSITY,
    "habitat_types": Metric.HABITAT_DIVERSITY,
    "temperature": Metric.TEMPERATURE, "temp": Metric.TEMPERATURE, "mean_temperature": Metric.TEMPERATURE,
    "rainfall": Metric.RAINFALL, "precipitation": Metric.RAINFALL, "annual_rainfall": Metric.RAINFALL,
    "pollution": Metric.POLLUTION, "pesticide_use": Metric.POLLUTION, "chemical_use": Metric.POLLUTION,
    "deforestation": Metric.DEFORESTATION, "tree_cover_loss": Metric.DEFORESTATION,
    "forest_loss": Metric.DEFORESTATION,
    "region": Metric.REGION, "location": Metric.REGION, "area": Metric.REGION,
    "climate_zone": Metric.CLIMATE_ZONE, "climate": Metric.CLIMATE_ZONE,
    "farm_size": Metric.FARM_SIZE, "area_ha": Metric.FARM_SIZE, "size": Metric.FARM_SIZE,
    "irrigation": Metric.IRRIGATION, "water_source": Metric.IRRIGATION,
}

REGION_HINTS = [
    "india", "kenya", "brazil", "australia", "spain", "france", "china", "nigeria",
    "ethiopia", "mexico", "usa", "united states", "canada", "indonesia", "vietnam",
    "punjab", "maharashtra", "rajasthan", "telangana", "karnataka", "sahel", "deccan",
    "andhra", "tamil nadu", "gujarat", "bihar", "kerala", "uttar pradesh", "madhya pradesh",
]


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def _normalise(text: str) -> str:
    t = text.lower()
    t = t.replace("%", " % ").replace("_", " ")
    t = re.sub(r"[\u2013\u2014]", "-", t)
    return re.sub(r"\s+", " ", t)


def extract_from_text(text: str) -> List[Observation]:
    """Pull every recognisable metric out of a free-text message."""
    if not text or not text.strip():
        return []
    raw = text
    t = _normalise(text)
    found: Dict[Metric, Observation] = {}

    # 1. Numeric patterns (first match wins per metric)
    for metric, pattern, unit in NUMERIC_PATTERNS:
        if metric in found:
            continue
        m = re.search(pattern, t)
        if not m:
            continue
        try:
            value = float(m.group(1))
        except (ValueError, IndexError):
            continue
        # Organic matter -> organic carbon via the conventional 1.724 factor.
        if unit == "%_om":
            value = round(value / 1.724, 2)
            unit = "%"
        if unit == "acre":
            value = round(value * 0.4047, 2)
            unit = "ha"
        if not _plausible(metric, value):
            continue
        found[metric] = Observation(
            metric=metric, raw=m.group(0).strip(), value=value, unit=unit, source="user"
        )

    # 2. Qualitative cues (longest phrase wins)
    for metric, cues in QUALITATIVE_CUES.items():
        if metric in found:
            continue
        best: Optional[Tuple[str, str]] = None
        for phrase, normalised in cues.items():
            if phrase in t and (best is None or len(phrase) > len(best[0])):
                best = (phrase, normalised)
        if best:
            found[metric] = Observation(
                metric=metric, raw=best[0], category=best[1], source="user"
            )

    # 3. Region
    if Metric.REGION not in found:
        for hint in REGION_HINTS:
            if re.search(rf"\b{re.escape(hint)}\b", t):
                found[Metric.REGION] = Observation(
                    metric=Metric.REGION, raw=hint, category=hint.title(), source="user"
                )
                break

    # 4. Climate zone implies a rainfall regime when rainfall is absent
    if Metric.RAINFALL not in found and Metric.CLIMATE_ZONE in found:
        zone = found[Metric.CLIMATE_ZONE].category
        implied = {"arid": "very low", "semi-arid": "low", "desert": "very low",
                   "humid": "high", "tropical": "high", "monsoon": "high"}.get(zone or "")
        if implied:
            found[Metric.RAINFALL] = Observation(
                metric=Metric.RAINFALL, raw=f"implied by '{zone}' climate",
                category=implied, source="default_assumption",
            )

    # 5. Disambiguation: "we cleared the forest" mentions forest but the land is
    # no longer forest. Drop the land-use reading rather than guess wrongly.
    if (
        Metric.DEFORESTATION in found
        and Metric.LAND_USE in found
        and str(found[Metric.LAND_USE].raw).strip() == "forest"
        and found[Metric.DEFORESTATION].category in {"moderate", "high", "severe"}
    ):
        del found[Metric.LAND_USE]

    _ = raw
    return [classify(o) for o in found.values()]


def _plausible(metric: Metric, value: float) -> bool:
    limits = {
        Metric.SOIL_PH: (0.0, 14.0),
        Metric.SOIL_ORGANIC_CARBON: (0.0, 60.0),
        Metric.SOIL_MOISTURE: (0.0, 100.0),
        Metric.TEMPERATURE: (-40.0, 60.0),
        Metric.RAINFALL: (0.0, 15000.0),
        Metric.SPECIES_RICHNESS: (0.0, 100000.0),
        Metric.HABITAT_DIVERSITY: (0.0, 1000.0),
        Metric.DEFORESTATION: (0.0, 100.0),
        Metric.FARM_SIZE: (0.0, 1000000.0),
    }
    lo, hi = limits.get(metric, (-1e12, 1e12))
    return lo <= value <= hi


def extract_from_json(payload) -> List[Observation]:
    """Accept a dict or a JSON string of structured site data."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("Structured input must be a JSON object.")

    # Allow a nested wrapper such as {"site": {...}} or {"metrics": {...}}
    flat: Dict[str, object] = {}
    for key, val in payload.items():
        if isinstance(val, dict) and key.lower() in {"site", "metrics", "data", "environment", "inputs"}:
            flat.update(val)
        else:
            flat[key] = val

    observations: List[Observation] = []
    for key, val in flat.items():
        metric = JSON_KEY_MAP.get(str(key).strip().lower().replace(" ", "_"))
        if metric is None or val is None or val == "":
            continue
        obs = _observation_from_value(metric, val)
        if obs:
            observations.append(classify(obs))
    return observations


def _observation_from_value(metric: Metric, val) -> Optional[Observation]:
    if isinstance(val, dict):
        unit = val.get("unit")
        val = val.get("value", val.get("val"))
        if val is None:
            return None
    else:
        unit = None

    if isinstance(val, bool):
        return Observation(metric=metric, raw=val, category=str(val).lower(), source="json")

    if isinstance(val, (int, float)):
        if not _plausible(metric, float(val)):
            return None
        default_units = {
            Metric.SOIL_PH: "pH", Metric.SOIL_ORGANIC_CARBON: "%",
            Metric.SOIL_MOISTURE: "% VWC", Metric.TEMPERATURE: "degC",
            Metric.RAINFALL: "mm/yr", Metric.SPECIES_RICHNESS: "species",
            Metric.HABITAT_DIVERSITY: "habitat types", Metric.DEFORESTATION: "%",
            Metric.FARM_SIZE: "ha",
        }
        return Observation(
            metric=metric, raw=val, value=float(val),
            unit=unit or default_units.get(metric), source="json",
        )

    text = str(val).strip()
    # A numeric string with a unit, e.g. "450 mm" or "0.3%"
    m = re.match(rf"^{NUM}\s*(.*)$", text)
    if m and m.group(1) is not None:
        try:
            numeric = float(m.group(1))
            if _plausible(metric, numeric):
                return Observation(
                    metric=metric, raw=text, value=numeric,
                    unit=unit or (m.group(2).strip() or None), source="json",
                )
        except ValueError:
            pass
    return Observation(metric=metric, raw=text, category=text.lower(), source="json")


def extract(user_input: str) -> List[Observation]:
    """Entry point: detects JSON input, otherwise parses as free text."""
    stripped = user_input.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return extract_from_json(stripped)
        except (json.JSONDecodeError, ValueError):
            pass
    return extract_from_text(user_input)
