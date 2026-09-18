"""
Dynamic clarification.

The system asks for the missing variable that would most change its answer, not
for every field it does not have. Priority is computed from the current state:
knowing rainfall is class is semi-arid makes soil moisture and land use urgent
(the dryland gate depends on them) while making temperature a nice-to-have.

The policy never asks about a metric already known, never asks more than
`max_questions` at once, and stops asking once enough variables are present to
reason with. That last condition is what stops the system interrogating the user
indefinitely.
"""

from __future__ import annotations

from typing import Dict, List

from core.reference import at_or_above, at_or_below
from core.schema import CORE_ENV_METRICS, ClarifyingQuestion, EnvState, Metric

MIN_METRICS_TO_REASON = 3
TARGET_METRICS = 5

# Baseline importance for reasoning coverage.
BASE_PRIORITY: Dict[Metric, float] = {
    Metric.SOIL_ORGANIC_CARBON: 5.5,
    Metric.LAND_USE: 5.0,
    Metric.RAINFALL: 4.5,
    Metric.SPECIES_RICHNESS: 3.5,
    Metric.SOIL_PH: 3.0,
    Metric.HABITAT_DIVERSITY: 2.5,
    Metric.POLLUTION: 2.5,
    Metric.SOIL_MOISTURE: 2.0,
    Metric.DEFORESTATION: 2.0,
    Metric.TEMPERATURE: 1.5,
}

QUESTION_TEXT: Dict[Metric, Dict[str, str]] = {
    Metric.SOIL_ORGANIC_CARBON: {
        "q": "What is your topsoil organic carbon, as a percentage? (If you only have organic matter %, give me that and I will convert it.)",
        "why": "Soil organic carbon sets how much water the profile retains, how much nutrient it cycles, and how much energy is available to the soil food web, so it constrains almost every other recommendation.",
        "eg": "e.g. 0.4% or 'organic matter 1.2%'",
    },
    Metric.LAND_USE: {
        "q": "What is the current land use - a single crop, a rotation, mixed cropping, grazing land, orchard, or degraded land?",
        "why": "Land use determines the structural habitat already present and which interventions are physically possible; land-use change is also the highest-ranked global driver of biodiversity loss.",
        "eg": "e.g. 'continuous monoculture wheat' or 'grazing land'",
    },
    Metric.RAINFALL: {
        "q": "What is the annual rainfall, in mm, or is it broadly low, moderate or high?",
        "why": "Rainfall sets whether water or nutrients limit biomass. In semi-arid conditions it changes the ordering of recommendations entirely, because carbon-building inputs are capped by available water.",
        "eg": "e.g. '450 mm' or 'low and erratic'",
    },
    Metric.SPECIES_RICHNESS: {
        "q": "What does biodiversity look like on the land - roughly how many species, or is it declining, moderate or rich?",
        "why": "This is the outcome variable being improved, and its current level distinguishes a site that needs habitat creation from one that needs existing habitat protected.",
        "eg": "e.g. 'declining, few pollinators' or '12 bird species'",
    },
    Metric.SOIL_PH: {
        "q": "What is the soil pH?",
        "why": "pH acts as a gate: below about 5.5, legume nitrogen fixation and root growth are impaired, so any legume-based recommendation would under-deliver until it is corrected.",
        "eg": "e.g. 5.4",
    },
    Metric.HABITAT_DIVERSITY: {
        "q": "What non-crop habitat exists on or next to the holding - hedgerows, tree lines, ponds, uncultivated margins, woodland?",
        "why": "Habitat diversity and connectivity act separately from habitat area; without knowing what is already there I cannot tell whether to add habitat or to link what exists.",
        "eg": "e.g. 'no hedgerows, one pond' or '3 habitat types'",
    },
    Metric.POLLUTION: {
        "q": "How heavy is chemical use or pollution pressure - pesticides, fertiliser, or any nearby industrial or nutrient runoff?",
        "why": "Chemical pressure caps the return on habitat measures, so if it is high the sequencing of recommendations changes: exposure has to come down before habitat creation pays off.",
        "eg": "e.g. 'routine insecticide sprays' or 'organic, no chemicals'",
    },
    Metric.SOIL_MOISTURE: {
        "q": "How moist is the soil through the season - persistently dry, adequate, or waterlogged? A volumetric % reading is ideal if you have one.",
        "why": "Moisture links rainfall to species survival: it determines whether the water that falls is actually retained where roots and soil organisms can use it.",
        "eg": "e.g. 'dry by mid-season' or '14% VWC'",
    },
    Metric.DEFORESTATION: {
        "q": "Has tree or woody cover been lost on or around the land recently, and roughly how much?",
        "why": "Recent clearance implies fragmentation, which needs connectivity measures rather than simply adding habitat area - a different intervention with different design rules.",
        "eg": "e.g. 'cleared about 30% five years ago' or 'no tree loss'",
    },
    Metric.TEMPERATURE: {
        "q": "What is the mean annual temperature, or is the climate broadly cool, warm or hot?",
        "why": "Temperature governs how fast added organic matter is mineralised, which changes both the expected size of a carbon gain and how much ongoing input is needed to hold it.",
        "eg": "e.g. '27 C' or 'hot'",
    },
}


def _contextual_bonus(metric: Metric, state: EnvState) -> float:
    """Raise the priority of metrics that would change the current diagnosis."""
    bonus = 0.0
    rain = state.classification(Metric.RAINFALL)
    soc = state.classification(Metric.SOIL_ORGANIC_CARBON)
    rich = state.classification(Metric.SPECIES_RICHNESS)
    lu = state.category(Metric.LAND_USE)
    defo = state.classification(Metric.DEFORESTATION)
    poll = state.classification(Metric.POLLUTION)

    dry = at_or_below(Metric.RAINFALL, rain, "semi_arid")
    low_carbon = at_or_below(Metric.SOIL_ORGANIC_CARBON, soc, "low")
    low_bio = at_or_below(Metric.SPECIES_RICHNESS, rich, "low")

    if dry and metric == Metric.SOIL_MOISTURE:
        bonus += 3.0            # the dryland water-first gate needs it
    if dry and metric == Metric.TEMPERATURE:
        bonus += 1.5            # evaporative demand matters in drylands
    if low_carbon and metric == Metric.SOIL_PH:
        bonus += 2.5            # decides whether legumes are viable
    if low_carbon and metric == Metric.LAND_USE:
        bonus += 1.5
    if low_bio and metric in (Metric.HABITAT_DIVERSITY, Metric.POLLUTION):
        bonus += 2.5            # distinguishes habitat deficit from chemical pressure
    if low_bio and metric == Metric.DEFORESTATION:
        bonus += 1.5            # distinguishes fragmentation from simple habitat loss
    if at_or_above(Metric.DEFORESTATION, defo, "moderate") and metric == Metric.HABITAT_DIVERSITY:
        bonus += 2.0
    if at_or_above(Metric.POLLUTION, poll, "high") and metric == Metric.SPECIES_RICHNESS:
        bonus += 2.0
    if lu in {"monoculture_cropland", "rotational_cropland"} and metric == Metric.SOIL_ORGANIC_CARBON:
        bonus += 1.5
    if lu == "grazing_land" and metric == Metric.RAINFALL:
        bonus += 1.5            # sets the rest period in rotational grazing

    # An observation the user never gave, that we assumed, is worth confirming.
    obs = state.get(metric)
    if obs and obs.source == "default_assumption":
        bonus += 2.0
    return bonus


def needs_clarification(state: EnvState) -> bool:
    return len(state.known_core_metrics()) < MIN_METRICS_TO_REASON


def next_questions(state: EnvState, max_questions: int = 2) -> List[ClarifyingQuestion]:
    """Rank missing metrics by how much they would change the answer."""
    known = set(state.known_core_metrics())
    candidates = []
    for metric in CORE_ENV_METRICS:
        if metric in known:
            continue
        score = BASE_PRIORITY.get(metric, 1.0) + _contextual_bonus(metric, state)
        candidates.append((score, metric))

    # Metrics that were assumed rather than supplied are also worth confirming.
    for metric in CORE_ENV_METRICS:
        obs = state.get(metric)
        if obs and obs.source == "default_assumption":
            candidates.append((BASE_PRIORITY.get(metric, 1.0) + 2.0, metric))

    candidates.sort(key=lambda x: (-x[0], x[1].value))

    seen, out = set(), []
    for score, metric in candidates:
        if metric in seen:
            continue
        seen.add(metric)
        text = QUESTION_TEXT.get(metric)
        if not text:
            continue
        out.append(
            ClarifyingQuestion(
                metric=metric,
                question=text["q"],
                why_it_matters=text["why"],
                example_answer=text["eg"],
                priority=round(score, 2),
            )
        )
        if len(out) >= max_questions:
            break
    return out


def coverage_note(state: EnvState) -> str:
    known = state.known_core_metrics()
    n = len(known)
    names = ", ".join(m.value.replace("_", " ") for m in known)
    if n < MIN_METRICS_TO_REASON:
        return f"{n} of the {MIN_METRICS_TO_REASON} variables needed to reason are available ({names or 'none yet'})."
    if n < TARGET_METRICS:
        return f"Reasoning over {n} variables ({names}). More detail would narrow the uncertainty."
    return f"Reasoning over {n} variables ({names})."
