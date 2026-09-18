"""
Multi-metric environmental reasoning engine.

This module, not the language model, decides what to recommend.

Each rule encodes a causal pattern that requires **several environmental
variables to hold simultaneously**. A rule that could fire on one variable alone
is not a rule here - the challenge explicitly asks for reasoning that combines
variables, so every rule declares the metrics it inspected and the cross-variable
interactions it relied on, and both are surfaced in the output.

Two rule categories:

  * `GATE` rules encode preconditions. A gate can suppress or downgrade another
    recommendation (acid soil blocks legume nitrogen fixation; heavy pesticide
    load caps the return on habitat creation). Gates are why this system can say
    "do not do the obvious thing yet, do this first" - which is where
    non-obvious recommendations come from.

  * `ACTION` rules propose interventions.

Every rule attaches `claim_ids`. The evidence layer then retrieves the passage
backing each claim. Claims whose evidence cannot be retrieved are dropped and
the recommendation's confidence falls, so the system degrades honestly instead
of asserting unsupported numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import yaml
from pathlib import Path

from core.reference import at_or_above, at_or_below, land_use_attributes
from core.schema import (
    Confidence,
    Direction,
    EnvState,
    Metric,
    MetricImpact,
    Recommendation,
    TimeHorizon,
)

CLAIMS_FILE = Path(__file__).resolve().parent.parent / "knowledge" / "claims.yaml"

_CLAIMS_CACHE: Optional[Dict[str, Dict]] = None


def load_claims() -> Dict[str, Dict]:
    global _CLAIMS_CACHE
    if _CLAIMS_CACHE is None:
        with open(CLAIMS_FILE, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        _CLAIMS_CACHE = {c["id"]: c for c in data["claims"]}
    return _CLAIMS_CACHE


# --------------------------------------------------------------------------
# Rule infrastructure
# --------------------------------------------------------------------------

@dataclass
class RuleOutcome:
    triggered_by: List[str]
    metrics_used: List[Metric]
    interactions: List[str]
    claim_ids: List[str]
    action: str
    mechanism: str
    caveats: List[str] = field(default_factory=list)
    priority_bonus: float = 0.0


@dataclass
class Rule:
    id: str
    title: str
    kind: str                       # "ACTION" | "GATE"
    horizon: TimeHorizon
    base_priority: float
    condition: Callable[[EnvState], Optional[RuleOutcome]]
    base_confidence: Confidence = Confidence.MEDIUM
    suppresses: List[str] = field(default_factory=list)
    requires_min_metrics: int = 3


CROPLAND_CLASSES = {
    "monoculture_cropland",
    "rotational_cropland",
    "mixed_cropland",
    "fallow",
    "orchard_plantation",
}


# --------------------------------------------------------------------------
# GATE rules - preconditions that reorder or block obvious advice
# --------------------------------------------------------------------------

def _gate_acid_soil(s: EnvState) -> Optional[RuleOutcome]:
    ph = s.classification(Metric.SOIL_PH)
    if not at_or_below(Metric.SOIL_PH, ph, "strongly_acidic"):
        return None
    soc = s.classification(Metric.SOIL_ORGANIC_CARBON)
    if soc is None:
        return None
    metrics = [Metric.SOIL_PH, Metric.SOIL_ORGANIC_CARBON]
    trig = [
        f"Soil pH is {s.get(Metric.SOIL_PH).display()} ({ph}), below the ~5.5 threshold where aluminium becomes soluble",
        f"Soil organic carbon is {soc}, so a legume-based carbon strategy would otherwise be the default advice",
    ]
    if s.has(Metric.RAINFALL):
        metrics.append(Metric.RAINFALL)
        trig.append(f"Rainfall class {s.classification(Metric.RAINFALL)} influences leaching and lime reaction rate")
    return RuleOutcome(
        triggered_by=trig,
        metrics_used=metrics,
        interactions=[
            "soil pH -> aluminium solubility -> root growth -> accessible soil water volume",
            "soil pH -> rhizobial symbiosis -> legume N fixation -> carbon stabilisation efficiency",
        ],
        claim_ids=["ph_gates_legumes", "legume_nitrogen_efficiency"],
        action=(
            "Correct soil acidity before investing in legume-based carbon or biodiversity measures. "
            "Apply an agricultural liming material at a rate set by a buffer-pH or lime-requirement test "
            "on your own soil, incorporate it, and re-test after one full season. Where lime is "
            "unavailable, target acid-tolerant legume species and cultivars instead."
        ),
        mechanism=(
            "Below roughly pH 5.5 aluminium becomes soluble and damages root apical meristems, phosphorus is "
            "fixed into unavailable forms, and rhizobial symbiosis is impaired. A legume cover crop sown onto "
            "this soil will nodulate poorly, fix far less nitrogen than expected, and therefore fail to supply "
            "the nitrogen needed to stabilise added carbon into soil organic matter. Sequencing matters: the "
            "same intervention that under-delivers now becomes effective once pH is corrected."
        ),
        caveats=[
            "Lime requirement depends on soil texture, buffering capacity and lime quality, so the rate must come from a local test rather than a generic figure.",
            "Liming raises pH slowly, typically over one to two seasons depending on particle fineness and incorporation.",
        ],
        priority_bonus=3.0,
    )


def _gate_pollution_cap(s: EnvState) -> Optional[RuleOutcome]:
    poll = s.classification(Metric.POLLUTION)
    if not at_or_above(Metric.POLLUTION, poll, "high"):
        return None
    rich = s.classification(Metric.SPECIES_RICHNESS)
    if rich is None:
        return None
    metrics = [Metric.POLLUTION, Metric.SPECIES_RICHNESS]
    trig = [
        f"Pollution pressure is classified {poll}",
        f"Species richness is {rich}, and habitat measures alone will not lift it while exposure stays high",
    ]
    if s.has(Metric.HABITAT_DIVERSITY):
        metrics.append(Metric.HABITAT_DIVERSITY)
        trig.append(f"Habitat diversity is {s.classification(Metric.HABITAT_DIVERSITY)}")
    if s.has(Metric.LAND_USE):
        metrics.append(Metric.LAND_USE)
    return RuleOutcome(
        triggered_by=trig,
        metrics_used=metrics,
        interactions=[
            "chemical exposure x habitat quality -> multiplicative, not additive, effect on populations",
            "herbicide use -> loss of flowering weeds -> forage supply -> pollinator abundance",
        ],
        claim_ids=["pollution_caps_habitat", "pesticide_sublethal", "pollution_driver_rank"],
        action=(
            "Reduce chemical exposure before, or at the same time as, creating habitat. Move to threshold-based "
            "integrated pest management: scout and spray on economic thresholds rather than calendar dates, "
            "drop prophylactic insecticide and seed treatments where no pest pressure is recorded, shift "
            "applications outside crop and margin flowering periods, and maintain an unsprayed buffer along any "
            "strip or margin you establish."
        ),
        mechanism=(
            "Habitat created inside a heavily treated matrix can act as an ecological trap: it attracts "
            "pollinators and natural enemies into an area of high exposure, so measured abundance rises briefly "
            "and then falls. Insecticides impose sub-lethal costs on foraging, navigation and reproduction even "
            "without visible mortality, and herbicides remove the flowering weeds that supply much of the forage "
            "in cropped land. The return on a flower strip is therefore capped by the chemical regime around it."
        ),
        caveats=[
            "Reducing prophylactic applications requires a monitoring routine to be in place first, or pest risk rises.",
        ],
        priority_bonus=3.5,
    )


def _gate_dryland_water_first(s: EnvState) -> Optional[RuleOutcome]:
    rain = s.classification(Metric.RAINFALL)
    if not at_or_below(Metric.RAINFALL, rain, "semi_arid"):
        return None
    soc = s.classification(Metric.SOIL_ORGANIC_CARBON)
    moist = s.classification(Metric.SOIL_MOISTURE)
    if soc is None and moist is None:
        return None
    metrics = [Metric.RAINFALL]
    trig = [f"Rainfall regime is {rain}, so water is the binding constraint on biomass production"]
    if soc:
        metrics.append(Metric.SOIL_ORGANIC_CARBON)
        trig.append(f"Soil organic carbon is {soc}, which itself reduces how much rainfall the profile retains")
    if moist:
        metrics.append(Metric.SOIL_MOISTURE)
        trig.append(f"Soil moisture is {moist}")
    if s.has(Metric.TEMPERATURE):
        metrics.append(Metric.TEMPERATURE)
        trig.append(f"Temperature class {s.classification(Metric.TEMPERATURE)} sets evaporative demand")
    if len(metrics) < 3:
        return None
    return RuleOutcome(
        triggered_by=trig,
        metrics_used=metrics,
        interactions=[
            "rainfall -> biomass production ceiling -> achievable carbon input (water gates carbon)",
            "soil organic carbon -> aggregation -> plant-available water capacity -> drought persistence",
            "temperature -> evaporative demand -> fraction of rainfall retained",
        ],
        claim_ids=["dryland_water_first", "residue_water_conservation", "soc_water_holding"],
        action=(
            "Sequence water capture ahead of carbon-building inputs. Install contour-aligned measures matched to "
            "your slope - contour bunds, trenches, planting pits or semi-circular basins - to convert runoff into "
            "infiltration, and keep the surface mulched with crop residue rather than bare between seasons. Only "
            "then scale up biomass-based carbon inputs."
        ),
        mechanism=(
            "In semi-arid and arid systems the amount of biomass that can be grown is capped by available water. "
            "Because carbon input to soil is biomass, attempting to raise soil carbon directly runs into a water "
            "ceiling. Capturing runoff and cutting evaporative loss raises stored soil water, which raises "
            "biomass, which raises carbon input - and the resulting organic matter improves aggregation and "
            "plant-available water capacity, feeding back into water retention. This ordering is the opposite of "
            "the usual 'add organic matter' advice and is what makes it effective here."
        ),
        caveats=[
            "Structure type and spacing depend on slope, soil depth and rainfall intensity; over-sized structures can concentrate flow and cause gullying.",
            "Residue retention competes with fodder and fuel uses where livestock or household demand is high.",
        ],
        priority_bonus=3.0,
    )


# --------------------------------------------------------------------------
# ACTION rules
# --------------------------------------------------------------------------

def _action_agroforestry(s: EnvState) -> Optional[RuleOutcome]:
    lu = s.category(Metric.LAND_USE) or s.classification(Metric.LAND_USE)
    if lu not in CROPLAND_CLASSES | {"degraded_land", "grazing_land"}:
        return None
    soc = s.classification(Metric.SOIL_ORGANIC_CARBON)
    rich = s.classification(Metric.SPECIES_RICHNESS)
    hab = s.classification(Metric.HABITAT_DIVERSITY)
    defo = s.classification(Metric.DEFORESTATION)
    rain = s.classification(Metric.RAINFALL)
    temp = s.classification(Metric.TEMPERATURE)

    structural_deficit = (
        at_or_below(Metric.HABITAT_DIVERSITY, hab, "low")
        or at_or_below(Metric.SPECIES_RICHNESS, rich, "low")
        or at_or_above(Metric.DEFORESTATION, defo, "moderate")
        or land_use_attributes(lu).get("structural_complexity") in {"very_low", "low"}
    )
    carbon_deficit = at_or_below(Metric.SOIL_ORGANIC_CARBON, soc, "low")
    if not (structural_deficit and carbon_deficit):
        return None

    metrics = [Metric.LAND_USE, Metric.SOIL_ORGANIC_CARBON]
    trig = [
        f"Land use is {lu}, whose structural complexity is "
        f"{land_use_attributes(lu).get('structural_complexity', 'low')}",
        f"Soil organic carbon is {soc}",
    ]
    interactions = [
        "woody perennial cover -> litter and root carbon input -> soil organic carbon",
        "vertical vegetation structure -> nesting, forage and refugia -> species richness",
    ]
    claims = ["af_biodiversity", "af_soc_cropland", "srccl_cobenefit_options"]

    if rich:
        metrics.append(Metric.SPECIES_RICHNESS)
        trig.append(f"Species richness is {rich}")
    if hab:
        metrics.append(Metric.HABITAT_DIVERSITY)
        trig.append(f"Habitat diversity is {hab}")
    if at_or_above(Metric.DEFORESTATION, defo, "moderate"):
        metrics.append(Metric.DEFORESTATION)
        trig.append(f"Recent tree cover loss is {defo}, so on-farm woody cover partially replaces what was removed")
        claims.append("af_landuse_buffer")
        interactions.append("tree removal -> lost connectivity and carbon -> partially recoverable via on-farm woody cover")
    if at_or_below(Metric.RAINFALL, rain, "semi_arid"):
        metrics.append(Metric.RAINFALL)
        trig.append(f"Rainfall is {rain}; agroforestry SOC response is proportionally largest in arid zones")
        claims.extend(["af_arid_response", "af_runoff_yield"])
        interactions.append("canopy and litter -> reduced runoff -> infiltration -> stored soil water -> species survival in dry spells")
    if at_or_above(Metric.TEMPERATURE, temp, "warm"):
        metrics.append(Metric.TEMPERATURE)
        trig.append(f"Temperature class is {temp}, so canopy shading provides thermal refugia as well as forage")
        claims.append("af_microclimate")
        interactions.append("canopy -> lower peak surface temperature and vapour pressure deficit -> microclimatic refugia")

    return RuleOutcome(
        triggered_by=trig,
        metrics_used=metrics,
        interactions=interactions,
        claim_ids=claims,
        action=(
            "Introduce woody perennials into the production system rather than alongside it: scattered "
            "parkland trees at low density, contour-aligned tree lines, or alley cropping with widely spaced "
            "hedgerow rows between cropped alleys. Favour nitrogen-fixing and locally native species, keep "
            "initial density low enough to avoid competing with the crop for water, and manage by pruning so "
            "light competition stays controllable."
        ),
        mechanism=(
            "Agroforestry is the intervention that addresses this site's carbon deficit and its structural habitat "
            "deficit through a single change, which is why it outranks measures that address only one. Trees add "
            "litter, root turnover and pruning residues, which is the carbon input term; they add vertical "
            "structure, perennial forage and nesting substrate, which is the habitat term; and canopy plus litter "
            "intercept rainfall energy and slow overland flow, raising the share of rainfall that infiltrates. "
            "Annual-crop diversification cannot substitute, because it changes temporal diversity without adding "
            "the structural template that most non-crop species require."
        ),
        caveats=[
            "Tree-crop competition for water and light is the main failure mode in dry systems; species choice, density and pruning regime determine whether the net effect is positive.",
            "Establishment takes several seasons, and biodiversity and carbon effects reported in the literature refer to established systems, not to the planting year.",
        ],
        priority_bonus=2.0,
    )


def _action_legume_cover_crop(s: EnvState) -> Optional[RuleOutcome]:
    lu = s.category(Metric.LAND_USE) or s.classification(Metric.LAND_USE)
    if lu not in CROPLAND_CLASSES:
        return None
    soc = s.classification(Metric.SOIL_ORGANIC_CARBON)
    if not at_or_below(Metric.SOIL_ORGANIC_CARBON, soc, "low"):
        return None
    ph = s.classification(Metric.SOIL_PH)
    # The acid-soil gate handles this case; do not also propose legumes.
    if at_or_below(Metric.SOIL_PH, ph, "strongly_acidic"):
        return None

    metrics = [Metric.LAND_USE, Metric.SOIL_ORGANIC_CARBON]
    trig = [f"Land use is {lu}", f"Soil organic carbon is {soc}, leaving substantial headroom for accumulation"]
    claims = ["cc_soc_rate", "cc_biodiversity", "legume_nitrogen_efficiency", "cc_runoff_erosion"]
    interactions = [
        "cover crop biomass -> carbon input -> soil organic carbon -> aggregate stability",
        "soil organic matter -> soil food web energy supply -> soil biodiversity -> nutrient mineralisation",
    ]
    if ph:
        metrics.append(Metric.SOIL_PH)
        trig.append(f"Soil pH is {ph}, which permits effective rhizobial nodulation")
    if s.has(Metric.SOIL_MOISTURE):
        metrics.append(Metric.SOIL_MOISTURE)
        trig.append(f"Soil moisture is {s.classification(Metric.SOIL_MOISTURE)}")
        interactions.append("soil organic carbon -> plant-available water capacity -> resilience to dry spells")
        claims.append("soc_water_holding")
    if at_or_above(Metric.TEMPERATURE, s.classification(Metric.TEMPERATURE), "warm"):
        metrics.append(Metric.TEMPERATURE)
        trig.append(f"Temperature is {s.classification(Metric.TEMPERATURE)}, so mineralisation is fast and inputs must be sustained")
        claims.append("warming_mineralisation")
    if s.has(Metric.SPECIES_RICHNESS):
        metrics.append(Metric.SPECIES_RICHNESS)
    if len(set(metrics)) < 3:
        return None

    return RuleOutcome(
        triggered_by=trig,
        metrics_used=metrics,
        interactions=interactions,
        claim_ids=claims,
        action=(
            "Sow a legume-dominant cover crop mixture into the fallow window rather than leaving the ground bare - "
            "for example a vetch or clover combined with a fast-establishing grass or cereal and a brassica for "
            "rooting depth. Terminate by rolling or mowing rather than incorporation where feasible, and leave the "
            "residue on the surface. Inoculate the legume seed if legumes have not been grown recently."
        ),
        mechanism=(
            "The mixture works through three linked routes. First, the added biomass is the carbon input that "
            "drives soil organic carbon accumulation. Second, biologically fixed nitrogen from the legume supplies "
            "the nitrogen that organic matter formation requires, so a larger fraction of the added carbon is "
            "stabilised rather than respired - a legume-free cover crop delivers less durable carbon per tonne of "
            "biomass. Third, surface residue and living roots maintain cover during the window when the soil would "
            "otherwise be bare, which is when runoff and erosion losses are highest. The resulting organic matter "
            "is also the energy base for the soil food web, so soil biological diversity rises with it."
        ),
        caveats=[
            "In water-limited systems a cover crop consumes soil water that the following cash crop would have used; terminate early and prioritise residue cover over maximum biomass.",
            "Reported sequestration rates are means across sites; the achievable rate at any single site depends on biomass produced, soil texture and temperature.",
        ],
        priority_bonus=1.5,
    )


def _action_habitat_strips(s: EnvState) -> Optional[RuleOutcome]:
    lu = s.category(Metric.LAND_USE) or s.classification(Metric.LAND_USE)
    if lu not in CROPLAND_CLASSES | {"grazing_land"}:
        return None
    rich = s.classification(Metric.SPECIES_RICHNESS)
    hab = s.classification(Metric.HABITAT_DIVERSITY)
    if not (
        at_or_below(Metric.SPECIES_RICHNESS, rich, "low")
        or at_or_below(Metric.HABITAT_DIVERSITY, hab, "low")
    ):
        return None

    metrics = [Metric.LAND_USE]
    trig = [f"Land use is {lu}"]
    if rich:
        metrics.append(Metric.SPECIES_RICHNESS)
        trig.append(f"Species richness is {rich}")
    if hab:
        metrics.append(Metric.HABITAT_DIVERSITY)
        trig.append(f"Habitat diversity is {hab}, indicating few distinct resource types on the holding")
    claims = ["fs_pest_control", "fs_distance_decay", "fs_edge_not_spillover", "forage_continuity", "pollinator_nesting"]
    interactions = [
        "habitat heterogeneity -> forage and nesting continuity -> pollinator and natural enemy populations",
        "natural enemy populations -> pest regulation -> reduced insecticide need -> lower chemical pressure on the same community",
    ]
    if s.has(Metric.POLLUTION):
        metrics.append(Metric.POLLUTION)
        trig.append(f"Pollution pressure is {s.classification(Metric.POLLUTION)}")
    if s.has(Metric.SOIL_ORGANIC_CARBON):
        metrics.append(Metric.SOIL_ORGANIC_CARBON)
    if s.has(Metric.HABITAT_DIVERSITY) and at_or_below(Metric.HABITAT_DIVERSITY, hab, "low"):
        claims.append("semi_natural_threshold")
    if len(set(metrics)) < 3:
        return None

    return RuleOutcome(
        triggered_by=trig,
        metrics_used=metrics,
        interactions=interactions,
        claim_ids=claims,
        action=(
            "Establish perennial, species-rich flower strips and margins rather than a single annual sowing. "
            "Use a mixture chosen so that something is in flower across the whole season, not only at one peak, "
            "and place strips so that field interiors are within reach of a margin - break large blocks with an "
            "internal strip instead of relying on the perimeter alone. Leave patches of bare, undisturbed ground "
            "and uncut stems over winter as nesting substrate, and target semi-natural cover of roughly 10-20% of "
            "the area."
        ),
        mechanism=(
            "Three design details drive most of the variance in whether this works. Service benefit falls off "
            "exponentially with distance from the planting, so strip placement determines how much of the field is "
            "actually reached. Effects strengthen with strip age and floral diversity, so a perennial mixture "
            "outperforms an annual monofloral sowing by a widening margin over years. And floral resources alone "
            "are insufficient, because many wild bee species are limited by nesting substrate - bare ground, dead "
            "wood, hollow stems - independently of forage, so a strip that supplies flowers while eliminating "
            "nesting habitat under-delivers. Expect a reliable gain in richness at the margin itself; in-field "
            "spillover is not consistent and should not be the justification."
        ),
        caveats=[
            "Evidence for margin plantings comes predominantly from northern hemisphere temperate systems; species mixes must be locally native and locally sourced.",
            "Effects on crop pollination and yield are variable, so this should be justified on conservation and pest-regulation grounds rather than on an expected yield gain.",
        ],
        priority_bonus=1.2,
    )


def _action_connectivity(s: EnvState) -> Optional[RuleOutcome]:
    defo = s.classification(Metric.DEFORESTATION)
    if not at_or_above(Metric.DEFORESTATION, defo, "moderate"):
        return None
    rich = s.classification(Metric.SPECIES_RICHNESS)
    hab = s.classification(Metric.HABITAT_DIVERSITY)
    if rich is None and hab is None:
        return None
    metrics = [Metric.DEFORESTATION]
    trig = [f"Tree cover loss is classified {defo}, so remaining habitat is likely fragmented"]
    if rich:
        metrics.append(Metric.SPECIES_RICHNESS)
        trig.append(f"Species richness is {rich}")
    if hab:
        metrics.append(Metric.HABITAT_DIVERSITY)
        trig.append(f"Habitat diversity is {hab}")
    if s.has(Metric.LAND_USE):
        metrics.append(Metric.LAND_USE)
        trig.append(f"Surrounding land use is {s.category(Metric.LAND_USE)}")
    if s.has(Metric.TEMPERATURE):
        metrics.append(Metric.TEMPERATURE)
    if len(set(metrics)) < 3:
        return None
    return RuleOutcome(
        triggered_by=trig,
        metrics_used=metrics,
        interactions=[
            "tree removal -> patch isolation -> local extinction without recolonisation -> declining richness",
            "connectivity -> dispersal capacity -> ability to track shifting climate envelopes",
        ],
        claim_ids=["fragmentation_connectivity", "connectivity_climate_tracking", "landuse_first_driver", "af_landuse_buffer"],
        action=(
            "Treat connectivity as a separate objective from habitat area. Map the remaining woody patches on and "
            "around the holding and link them with continuous linear features - hedgerows along boundaries, tree "
            "lines on contours, and wooded strips along any watercourse - rather than adding another isolated "
            "block. Protect existing mature trees and remnant patches first, since they are the recolonisation "
            "sources that make new planting effective."
        ),
        mechanism=(
            "Habitat area and habitat connectivity are non-substitutable variables. Small isolated patches lose "
            "species through local extinction that is never offset by recolonisation, so the same total area held "
            "in linked units supports more species than in scattered fragments. Connectivity also determines "
            "whether species can shift their ranges as climate changes: a population with no dispersal route "
            "cannot track a moving climate envelope regardless of how suitable the destination is. That is why "
            "planting the same number of trees in a line between remnants outperforms planting them in a block."
        ),
        caveats=[
            "Linear features benefit mobile taxa most; specialists of interior habitat need patch area, which corridors do not provide.",
            "Retaining existing mature trees is generally higher value per unit effort than new planting, and is irreversible if lost.",
        ],
        priority_bonus=2.2,
    )


def _action_rotation(s: EnvState) -> Optional[RuleOutcome]:
    lu = s.category(Metric.LAND_USE) or s.classification(Metric.LAND_USE)
    if lu != "monoculture_cropland":
        return None
    rich = s.classification(Metric.SPECIES_RICHNESS)
    soc = s.classification(Metric.SOIL_ORGANIC_CARBON)
    if rich is None and soc is None:
        return None
    metrics = [Metric.LAND_USE]
    trig = ["Land use is continuous monoculture cropland, the least diversified cropping configuration"]
    if rich:
        metrics.append(Metric.SPECIES_RICHNESS)
        trig.append(f"Species richness is {rich}")
    if soc:
        metrics.append(Metric.SOIL_ORGANIC_CARBON)
        trig.append(f"Soil organic carbon is {soc}")
    if s.has(Metric.POLLUTION):
        metrics.append(Metric.POLLUTION)
        trig.append(f"Pollution pressure is {s.classification(Metric.POLLUTION)}; rotation reduces the pest pressure that drives applications")
    if s.has(Metric.SOIL_MOISTURE):
        metrics.append(Metric.SOIL_MOISTURE)
    if len(set(metrics)) < 3:
        return None
    return RuleOutcome(
        triggered_by=trig,
        metrics_used=metrics,
        interactions=[
            "crop sequence diversity -> break in pest and pathogen cycles -> lower insecticide and fungicide need -> reduced chemical pressure on non-target fauna",
            "varied root architecture and residue chemistry -> varied substrate for soil biota -> soil biological diversity",
        ],
        claim_ids=["rotation_biodiversity", "diversification_overall", "intercropping_biodiversity"],
        action=(
            "Replace continuous monoculture with a rotation of at least three functionally different crops - "
            "including a legume and, where the season allows, a deep-rooting species - rather than alternating two "
            "crops of the same family. Sequence so that no crop follows itself and residue types differ between "
            "years."
        ),
        mechanism=(
            "Rotation delivers a substantially larger biodiversity gain than intercropping because it breaks pest "
            "and pathogen cycles across seasons rather than only varying what grows side by side within one. The "
            "resulting reduction in pest pressure lowers the need for insecticide, which removes a pressure on the "
            "same invertebrate community the intervention is meant to support - so the biodiversity benefit runs "
            "partly through the chemistry, not only through the plants. Varying root architecture and residue "
            "chemistry between years also diversifies the substrate available to soil organisms."
        ),
        caveats=[
            "Rotation delivers less than agroforestry for biodiversity because it adds temporal but not structural diversity; where both are feasible, structural change is the larger lever.",
            "Market access and equipment for alternative crops are often the real constraint rather than agronomy.",
        ],
        priority_bonus=1.0,
    )


def _action_grazing(s: EnvState) -> Optional[RuleOutcome]:
    lu = s.category(Metric.LAND_USE) or s.classification(Metric.LAND_USE)
    if lu != "grazing_land":
        return None
    soc = s.classification(Metric.SOIL_ORGANIC_CARBON)
    rich = s.classification(Metric.SPECIES_RICHNESS)
    if soc is None and rich is None:
        return None
    metrics = [Metric.LAND_USE]
    trig = ["Land use is grazing land, where stocking management is the dominant control on both vegetation and soil"]
    if soc:
        metrics.append(Metric.SOIL_ORGANIC_CARBON)
        trig.append(f"Soil organic carbon is {soc}")
    if rich:
        metrics.append(Metric.SPECIES_RICHNESS)
        trig.append(f"Species richness is {rich}")
    if s.has(Metric.RAINFALL):
        metrics.append(Metric.RAINFALL)
        trig.append(f"Rainfall is {s.classification(Metric.RAINFALL)}, which sets the recovery period vegetation needs after defoliation")
    if s.has(Metric.SOIL_MOISTURE):
        metrics.append(Metric.SOIL_MOISTURE)
    if len(set(metrics)) < 3:
        return None
    return RuleOutcome(
        triggered_by=trig,
        metrics_used=metrics,
        interactions=[
            "grazing pressure -> residual leaf area -> photosynthetic capacity -> root carbon input -> soil organic carbon",
            "rest period length -> flowering and seed set of palatable species -> plant diversity -> invertebrate diversity",
            "rainfall -> recovery rate -> rest period required before regrazing",
        ],
        claim_ids=["erosion_exceeds_formation", "srccl_cobenefit_options", "soc_soil_biodiversity", "diversification_overall"],
        action=(
            "Move from continuous grazing to planned rotational grazing with defined rest periods: subdivide into "
            "paddocks, graze each for a short period, and set the rest interval by observed regrowth rather than by "
            "the calendar - substantially longer in dry years. Set stocking so that a residual cover of vegetation "
            "always remains rather than grazing to bare ground, and leave some areas ungrazed through the flowering "
            "period each season on a rotating basis."
        ),
        mechanism=(
            "Continuous grazing removes leaf area faster than it regrows, which cuts photosynthetic capacity and "
            "therefore the root carbon input that builds soil organic matter, while selective repeated defoliation "
            "eliminates palatable species before they set seed and shifts the sward toward unpalatable and annual "
            "species. Rest periods restore leaf area, allow flowering and seed set, and maintain the ground cover "
            "that prevents erosion losses. Because recovery rate depends on water, the same stocking rate that is "
            "sustainable in a wet year degrades the sward in a dry one, so the rest interval must track rainfall."
        ),
        caveats=[
            "Rotational grazing requires fencing or herding labour and water points in each subdivision, which is often the binding constraint.",
            "Outcomes depend more on matching stocking rate and rest period to actual growth than on the rotation pattern itself.",
        ],
        priority_bonus=1.4,
    )


def _action_monitoring(s: EnvState) -> Optional[RuleOutcome]:
    """Always-available low-priority rule: how to verify any of the above."""
    if len(s.known_core_metrics()) < 3:
        return None
    metrics = s.known_core_metrics()[:4]
    return RuleOutcome(
        triggered_by=[f"{len(s.known_core_metrics())} environmental variables are characterised, so a baseline can be fixed now"],
        metrics_used=metrics,
        interactions=[
            "slow-responding stock variables (SOC) vs fast-responding process indicators (infiltration, aggregate stability, earthworm counts)",
        ],
        claim_ids=["soc_monitoring_lag", "soc_reversibility"],
        action=(
            "Fix a baseline before changing management, and monitor fast proxies rather than waiting on soil carbon. "
            "Sample soil at a fixed depth increment with bulk density correction at georeferenced points, and repeat "
            "on the same points every three to five years. In between, track infiltration time, aggregate stability, "
            "earthworm counts and ground cover annually, plus a simple fixed-transect count of flowering plants and "
            "pollinators at the same date each season."
        ),
        mechanism=(
            "Soil organic carbon changes slowly against high spatial variability, so a measurable difference "
            "typically needs several years and a consistent protocol - monitoring it annually produces noise that "
            "is easily misread as failure. Aggregate stability, infiltration rate and soil fauna respond within one "
            "to two seasons and move in the same direction, so they serve as early indicators. Baseline points must "
            "be georeferenced because relocating sampling points between rounds introduces more variance than the "
            "management effect being measured."
        ),
        caveats=[
            "Without a pre-intervention baseline, later change cannot be attributed to the intervention.",
        ],
        priority_bonus=0.0,
    )


# --------------------------------------------------------------------------
# Rule registry
# --------------------------------------------------------------------------

RULES: List[Rule] = [
    Rule("gate_pollution", "Reduce chemical exposure before creating habitat", "GATE",
         TimeHorizon.SHORT, 9.0, _gate_pollution_cap, Confidence.HIGH,
         suppresses=[]),
    Rule("gate_acid_soil", "Correct soil acidity before legume-based measures", "GATE",
         TimeHorizon.SHORT, 8.5, _gate_acid_soil, Confidence.HIGH,
         suppresses=["action_legume_cover_crop"]),
    Rule("gate_dryland_water", "Capture water before building carbon", "GATE",
         TimeHorizon.SHORT, 8.0, _gate_dryland_water_first, Confidence.HIGH),
    Rule("action_agroforestry", "Integrate woody perennials into the production system", "ACTION",
         TimeHorizon.LONG, 7.0, _action_agroforestry, Confidence.HIGH),
    Rule("action_connectivity", "Restore connectivity between remaining habitat patches", "ACTION",
         TimeHorizon.LONG, 6.5, _action_connectivity, Confidence.HIGH),
    Rule("action_legume_cover_crop", "Legume-based cover cropping in the fallow window", "ACTION",
         TimeHorizon.MEDIUM, 6.0, _action_legume_cover_crop, Confidence.HIGH),
    Rule("action_habitat_strips", "Perennial species-rich field margins and flower strips", "ACTION",
         TimeHorizon.MEDIUM, 5.5, _action_habitat_strips, Confidence.HIGH),
    Rule("action_grazing", "Planned rotational grazing with rainfall-matched rest", "ACTION",
         TimeHorizon.MEDIUM, 5.0, _action_grazing, Confidence.MEDIUM),
    Rule("action_rotation", "Diversify the crop sequence", "ACTION",
         TimeHorizon.MEDIUM, 4.5, _action_rotation, Confidence.HIGH),
    Rule("action_monitoring", "Establish a baseline and monitor fast proxies", "ACTION",
         TimeHorizon.SHORT, 1.0, _action_monitoring, Confidence.HIGH),
]


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

CONFIDENCE_ORDER = [Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH]


def _downgrade(c: Confidence, steps: int = 1) -> Confidence:
    idx = max(0, CONFIDENCE_ORDER.index(c) - steps)
    return CONFIDENCE_ORDER[idx]


def _build_impacts(claim_ids: List[str]) -> List[MetricImpact]:
    claims = load_claims()
    impacts: List[MetricImpact] = []
    seen = set()
    for cid in claim_ids:
        claim = claims.get(cid)
        if not claim:
            continue
        try:
            metric = Metric(claim["metric"])
        except ValueError:
            continue
        key = (metric, claim["direction"])
        if key in seen:
            continue
        seen.add(key)
        impacts.append(
            MetricImpact(
                metric=metric,
                direction=Direction(claim["direction"]),
                magnitude=claim["magnitude"],
                horizon=TimeHorizon(claim["horizon"]),
                basis=cid,
            )
        )
    return impacts


def reason(state: EnvState) -> List[Recommendation]:
    """Run the rule graph over the environmental state."""
    claims = load_claims()
    fired: List[tuple] = []
    suppressed: set = set()

    for rule in RULES:
        outcome = rule.condition(state)
        if outcome is None:
            continue
        if len(set(outcome.metrics_used)) < rule.requires_min_metrics and rule.id != "action_monitoring":
            continue
        fired.append((rule, outcome))
        suppressed.update(rule.suppresses)

    recommendations: List[Recommendation] = []
    for rule, outcome in fired:
        if rule.id in suppressed:
            continue

        valid_claims = [cid for cid in outcome.claim_ids if cid in claims]
        dropped = [cid for cid in outcome.claim_ids if cid not in claims]

        confidence = rule.base_confidence
        reasons: List[str] = []
        uncertainty: List[str] = []

        n_metrics = len(set(outcome.metrics_used))
        reasons.append(f"reasoned over {n_metrics} environmental variables")

        if dropped:
            confidence = _downgrade(confidence)
            uncertainty.append(
                f"{len(dropped)} supporting claim(s) could not be resolved in the knowledge base and were excluded."
            )

        # Assumed values weaken confidence.
        assumed = [
            m.value.replace("_", " ")
            for m in outcome.metrics_used
            if state.get(m) and state.get(m).source == "default_assumption"
        ]
        if assumed:
            confidence = _downgrade(confidence)
            uncertainty.append(f"Assumed rather than supplied: {', '.join(sorted(set(assumed)))}.")
            reasons.append("some inputs were assumed")

        strengths = [claims[c].get("strength") for c in valid_claims]
        if strengths and all(s in {"synthesis", "moderate"} for s in strengths):
            confidence = _downgrade(confidence)
            uncertainty.append("Supporting evidence is synthesis-level rather than direct meta-analytic measurement.")
        elif any(s == "high" for s in strengths):
            reasons.append("supported by meta-analytic or major assessment evidence")

        if n_metrics >= 5:
            reasons.append("multiple independent variables agree on the diagnosis")

        recommendations.append(
            Recommendation(
                intervention_id=rule.id,
                title=rule.title,
                action=outcome.action,
                mechanism=outcome.mechanism,
                triggered_by=outcome.triggered_by,
                metrics_used=list(dict.fromkeys(outcome.metrics_used)),
                impacts=_build_impacts(valid_claims),
                interactions=outcome.interactions,
                time_horizon=rule.horizon,
                confidence=confidence,
                confidence_reason="; ".join(reasons),
                uncertainty=uncertainty,
                caveats=outcome.caveats,
                claim_ids=valid_claims,
                priority=rule.base_priority + outcome.priority_bonus + 0.3 * n_metrics,
            )
        )

    recommendations.sort(key=lambda r: r.priority, reverse=True)
    return recommendations
