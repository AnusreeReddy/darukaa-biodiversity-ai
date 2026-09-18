"""
Canonical data structures for the Darukaa.Earth biodiversity intelligence system.

Everything that moves between modules is one of these objects. The reasoning
engine never sees free text, and the synthesiser never sees anything that is not
backed by an EvidenceItem.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Environmental variables tracked by the system
# --------------------------------------------------------------------------

class Metric(str, Enum):
    """The environmental variables named in the Darukaa.Earth challenge."""

    SOIL_PH = "soil_ph"
    SOIL_ORGANIC_CARBON = "soil_organic_carbon"      # % (w/w) topsoil
    SOIL_MOISTURE = "soil_moisture"                  # qualitative or % VWC
    LAND_USE = "land_use"                            # categorical
    SPECIES_RICHNESS = "species_richness"            # count or qualitative
    HABITAT_DIVERSITY = "habitat_diversity"          # qualitative / count
    TEMPERATURE = "temperature"                      # deg C, mean annual
    RAINFALL = "rainfall"                            # mm/yr or qualitative
    POLLUTION = "pollution" 
    PEST_CONTROL_SERVICE = "pest_control_service"                         # categorical/qualitative
    DEFORESTATION = "deforestation"                  # qualitative / % loss

    # Contextual (not scored variables, but drive reasoning)
    REGION = "region"
    CLIMATE_ZONE = "climate_zone"
    FARM_SIZE = "farm_size"
    IRRIGATION = "irrigation"


NUMERIC_METRICS = {
    Metric.SOIL_PH,
    Metric.SOIL_ORGANIC_CARBON,
    Metric.SOIL_MOISTURE,
    Metric.TEMPERATURE,
    Metric.RAINFALL,
    Metric.SPECIES_RICHNESS,
    Metric.FARM_SIZE,
}

# The variables that count toward the "at least 3 environmental variables"
# requirement in the challenge brief.
CORE_ENV_METRICS = [
    Metric.SOIL_PH,
    Metric.SOIL_ORGANIC_CARBON,
    Metric.SOIL_MOISTURE,
    Metric.LAND_USE,
    Metric.SPECIES_RICHNESS,
    Metric.HABITAT_DIVERSITY,
    Metric.TEMPERATURE,
    Metric.RAINFALL,
    Metric.POLLUTION,
    Metric.DEFORESTATION,
]


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TimeHorizon(str, Enum):
    SHORT = "short"      # 0-2 years
    MEDIUM = "medium"    # 2-5 years
    LONG = "long"        # 5+ years


class Direction(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    STABILISE = "stabilise"


# --------------------------------------------------------------------------
# Observations and state
# --------------------------------------------------------------------------

class Observation(BaseModel):
    """A single measured or reported value for one metric."""

    metric: Metric
    raw: Any = Field(description="Value exactly as supplied by the user.")
    value: Optional[float] = Field(
        default=None, description="Parsed numeric value, when the metric is numeric."
    )
    unit: Optional[str] = None
    category: Optional[str] = Field(
        default=None, description="Normalised categorical value, e.g. 'monoculture_cropland'."
    )
    classification: Optional[str] = Field(
        default=None,
        description="Interpretation from the reference layer, e.g. 'very_low', 'acidic'.",
    )
    source: str = Field(
        default="user", description="user | json | geo_lookup | default_assumption"
    )
    turn: int = 0

    def display(self) -> str:
        if self.value is not None:
            unit = f" {self.unit}" if self.unit else ""
            return f"{self.value:g}{unit}"
        return str(self.category or self.raw)


class EnvState(BaseModel):
    """Accumulated environmental picture across all conversation turns."""

    observations: Dict[str, Observation] = Field(default_factory=dict)
    turn: int = 0
    notes: List[str] = Field(default_factory=list)

    # -- access helpers ---------------------------------------------------
    def set(self, obs: Observation) -> None:
        obs.turn = self.turn
        self.observations[obs.metric.value] = obs

    def get(self, metric: Metric) -> Optional[Observation]:
        return self.observations.get(metric.value)

    def value(self, metric: Metric) -> Optional[float]:
        obs = self.get(metric)
        return obs.value if obs else None

    def category(self, metric: Metric) -> Optional[str]:
        obs = self.get(metric)
        if not obs:
            return None
        return obs.category

    def classification(self, metric: Metric) -> Optional[str]:
        obs = self.get(metric)
        if not obs:
            return None
        return obs.classification

    def has(self, metric: Metric) -> bool:
        return metric.value in self.observations

    def known_core_metrics(self) -> List[Metric]:
        return [m for m in CORE_ENV_METRICS if self.has(m)]

    def missing_core_metrics(self) -> List[Metric]:
        return [m for m in CORE_ENV_METRICS if not self.has(m)]

    def summary_lines(self) -> List[str]:
        out = []
        for metric in CORE_ENV_METRICS + [Metric.REGION, Metric.CLIMATE_ZONE, Metric.IRRIGATION]:
            obs = self.get(metric)
            if obs is None:
                continue
            label = metric.value.replace("_", " ").title()
            cls = f" ({obs.classification})" if obs.classification else ""
            out.append(f"{label}: {obs.display()}{cls}")
        return out


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    """A retrieved passage plus its provenance. Nothing is asserted without one."""

    chunk_id: str
    text: str
    source_id: str
    title: str
    organisation: str
    year: Optional[int] = None
    url: Optional[str] = None
    citation: str = ""
    section: Optional[str] = None
    score: float = 0.0
    retrieval: str = "hybrid"   # vector | bm25 | hybrid

    def short_ref(self) -> str:
        yr = f" {self.year}" if self.year else ""
        return f"{self.organisation}{yr}"


class MetricImpact(BaseModel):
    metric: Metric
    direction: Direction
    magnitude: Optional[str] = Field(
        default=None, description="Quantified expectation, verbatim from the evidence base."
    )
    horizon: TimeHorizon = TimeHorizon.MEDIUM
    basis: Optional[str] = Field(default=None, description="claim_id supporting this impact.")


class Recommendation(BaseModel):
    """One actionable intervention, fully traceable."""

    intervention_id: str
    title: str
    action: str = Field(description="WHAT to do, concretely.")
    mechanism: str = Field(description="WHY it works - the causal chain.")
    triggered_by: List[str] = Field(
        default_factory=list,
        description="Human-readable conditions from the state that fired this rule.",
    )
    metrics_used: List[Metric] = Field(
        default_factory=list, description="Variables the rule actually reasoned over."
    )
    impacts: List[MetricImpact] = Field(default_factory=list)
    interactions: List[str] = Field(
        default_factory=list,
        description="Cross-variable couplings made explicit, e.g. 'SOC -> water holding capacity -> species survival'.",
    )
    time_horizon: TimeHorizon = TimeHorizon.MEDIUM
    confidence: Confidence = Confidence.MEDIUM
    confidence_reason: str = ""
    uncertainty: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    claim_ids: List[str] = Field(default_factory=list)
    priority: float = 0.0

    def n_metrics(self) -> int:
        return len(set(self.metrics_used))


class ClarifyingQuestion(BaseModel):
    metric: Metric
    question: str
    why_it_matters: str
    example_answer: str = ""
    priority: float = 0.0


class SystemResponse(BaseModel):
    """What the app renders for one user turn."""

    mode: str  # "clarify" | "recommend" | "chat"
    state: EnvState
    questions: List[ClarifyingQuestion] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    interpretation: List[str] = Field(default_factory=list)
    message: str = ""
    coverage_note: str = ""

    def to_json_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")
