# Darukaa.Earth — AI Biodiversity Intelligence

An environmental reasoning system that diagnoses land from partial information, reasons across
multiple environmental variables simultaneously, and produces interventions that are supported by
passages retrieved from an indexed scientific knowledge base.

It is deliberately **not** an LLM with documents attached. The recommendations are produced by an
explicit, inspectable rule graph over classified environmental variables. Retrieval supplies the
evidence for each claim, and a claim whose evidence cannot be retrieved is dropped rather than
asserted.

```
python -m knowledge.ingest && streamlit run app.py
```

---

## Table of contents

1. [What it does](#1-what-it-does)
2. [Why this design](#2-why-this-design)
3. [Architecture](#3-architecture)
4. [Data schema](#4-data-schema)
5. [Knowledge layer and data sources](#5-knowledge-layer-and-data-sources)
6. [RAG pipeline](#6-rag-pipeline)
7. [The reasoning engine](#7-the-reasoning-engine)
8. [Conversational intelligence](#8-conversational-intelligence)
9. [Output contract](#9-output-contract)
10. [Local setup](#10-local-setup)
11. [Evaluation and tests](#11-evaluation-and-tests)
12. [Deployment](#12-deployment)
13. [Honest limitations](#13-honest-limitations)
14. [Repository map](#14-repository-map)

---

## 1. What it does

A worked example, taken from the challenge brief.

**Input**

```
Soil organic carbon 0.3%, rainfall low, monoculture wheat, semi-arid region, 28 C
```

**What the system does**

1. Parses five variables, including a climate zone and a temperature.
2. Classifies each against reference thresholds: SOC `critically_low`, rainfall `semi_arid`,
   land use `monoculture_cropland`, temperature `hot`.
3. Runs the rule graph. Three rules fire, each requiring several variables at once.
4. Retrieves supporting passages for every claim those rules make.
5. Returns ranked recommendations with mechanism, impacted metrics, horizon, confidence and sources.

**Top recommendation returned**

> **Capture water before building carbon** — install contour-aligned measures matched to slope and
> keep the surface mulched, *before* scaling biomass-based carbon inputs.
>
> Why: in semi-arid systems the biomass that can be grown is capped by available water, and carbon
> input *is* biomass, so raising soil carbon directly runs into a water ceiling.

This is the point of the system. The obvious answer to "low soil carbon" is "add organic matter".
The system instead recognises that rainfall gates the intervention and **reorders the advice**. That
reordering comes from a rule that inspects rainfall, SOC and temperature together — not from
prompting.

### Non-obvious behaviours worth checking in the demo

| Enter this | What happens | Why it matters |
|---|---|---|
| `pH 5.1, SOC 0.8%, monoculture maize, rainfall 900mm, biodiversity declining` | Legume cover cropping is **suppressed**; liming is surfaced first | Below pH 5.5 rhizobial symbiosis fails, so the standard advice would under-deliver |
| then `Actually the pH is 6.9` | Legume recommendation **returns** | Memory holds a correction and revises the diagnosis |
| `We spray insecticide weekly, biodiversity declining, no field margins, monoculture cotton, SOC 1.1%` | Exposure reduction ranks **above** flower strips | Habitat inside a treated matrix can act as an ecological trap |
| `Biodiversity is declining on my land` | Asks two targeted questions, gives no advice | Below three variables the system will not produce single-variable answers |

---

## 2. Why this design

The challenge asks for depth of reasoning, scientific grounding and a retrievable knowledge layer,
and explicitly rules out generic LLM solutions. Three design decisions follow from that.

**Reasoning is deterministic code, not prompting.** The rule graph in `core/reasoning.py` is
readable: a reviewer can open it and see exactly which variables produce which recommendation and
why. It is unit-tested, reproducible, and cannot drift between runs.

**Retrieval is used for evidence, not for answers.** Rules emit `claim_id`s. The evidence layer
retrieves passages for each claim, restricted to the sources the claim declares. Claims that
retrieve nothing are removed, their metric impacts are removed with them, and confidence is
downgraded. A quantified effect can therefore never appear without an indexed passage behind it.

**The system runs offline and deterministically.** Embeddings are fitted locally, there is no API
key and no model download. Every run of the evaluation produces identical results on any machine,
which is what makes the reported numbers meaningful.

---

## 3. Architecture

```
                        user text  or  structured JSON
                                     │
                    ┌────────────────▼────────────────┐
                    │  core/extract.py                │  deterministic parsing:
                    │  patterns + controlled vocab    │  numbers, units, phrases, JSON keys
                    └────────────────┬────────────────┘
                                     │  Observation[]
                    ┌────────────────▼────────────────┐
                    │  core/reference.py              │  STRUCTURED DATA LAYER
                    │  thresholds.json                │  0.3 % ──▶ "critically_low"
                    └────────────────┬────────────────┘  400 mm ──▶ "semi_arid"
                                     │  classified
                    ┌────────────────▼────────────────┐
                    │  core/conversation.py           │  MEMORY
                    │  merge · correct · accumulate   │  state persists across turns
                    └────────┬───────────────┬────────┘
                             │               │
              < 3 variables  │               │  >= 3 variables
                    ┌────────▼────────┐      │
                    │ core/clarify.py │      │        asks the variable that would
                    │ ranked question │      │        most change the answer
                    └─────────────────┘      │
                                   ┌─────────▼─────────┐
                                   │ core/reasoning.py │  MULTI-METRIC RULE GRAPH
                                   │ GATE + ACTION     │  gates can suppress actions
                                   └─────────┬─────────┘
                                             │  Recommendation[] + claim_ids
                                   ┌─────────▼─────────┐
                                   │ core/evidence.py  │  claim ──▶ retrieval
                                   └─────────┬─────────┘  unsupported ──▶ dropped
                                             │
                    ┌────────────────────────▼────────────────────────┐
                    │  core/retrieve.py    HYBRID RETRIEVAL           │
                    │  Chroma (dense LSA)  +  BM25   fused via RRF    │
                    └────────────────────────┬────────────────────────┘
                                             │  EvidenceItem[]
                                   ┌─────────▼─────────┐
                                   │  core/render.py   │  markdown + JSON
                                   └─────────┬─────────┘
                                      app.py │ cli.py
```

Every arrow carries a typed Pydantic object (`core/schema.py`). The reasoning engine never sees free
text, and the renderer never sees an unsupported claim.

---

## 4. Data schema

### 4.1 Runtime objects (`core/schema.py`)

| Object | Purpose | Key fields |
|---|---|---|
| `Observation` | One reported value for one metric | `metric`, `raw`, `value`, `unit`, `category`, `classification`, `source`, `turn` |
| `EnvState` | Accumulated site picture across turns | `observations: {metric → Observation}`, `turn` |
| `EvidenceItem` | A retrieved passage with provenance | `chunk_id`, `text`, `source_id`, `title`, `organisation`, `year`, `url`, `citation`, `section`, `score`, `retrieval` |
| `MetricImpact` | Expected effect on one metric | `metric`, `direction`, `magnitude`, `horizon`, `basis` (claim id) |
| `Recommendation` | One intervention, fully traceable | `action`, `mechanism`, `triggered_by`, `metrics_used`, `impacts`, `interactions`, `time_horizon`, `confidence`, `uncertainty`, `caveats`, `evidence`, `claim_ids` |
| `ClarifyingQuestion` | One targeted question | `metric`, `question`, `why_it_matters`, `example_answer`, `priority` |
| `SystemResponse` | One turn of output | `mode`, `state`, `questions`, `recommendations`, `interpretation`, `coverage_note` |

`Observation.source` is one of `user`, `json`, `default_assumption` — assumed values are flagged
in the output and reduce confidence.

### 4.2 Environmental variables tracked

All ten variables named in the challenge, plus four contextual ones.

`soil_ph` · `soil_organic_carbon` · `soil_moisture` · `land_use` · `species_richness` ·
`habitat_diversity` · `temperature` · `rainfall` · `pollution` · `deforestation`
— contextual: `region` · `climate_zone` · `farm_size` · `irrigation`

### 4.3 Storage

No database server is required; the system uses three file-backed stores.

| Store | Location | Format | Contents |
|---|---|---|---|
| Reference thresholds | `knowledge/reference_data/thresholds.json` | JSON | Classification bands per metric, each with a `source_id` |
| Source registry | `knowledge/sources.yaml` | YAML | 14 sources with full citation metadata |
| Claims registry | `knowledge/claims.yaml` | YAML | 38 quantified claims → metric, magnitude, horizon, sources, retrieval query |
| Vector store | `knowledge/vectorstore/chroma/` | ChromaDB (persistent) | 80 chunks with embeddings and metadata |
| Lexical index | `knowledge/vectorstore/bm25.pkl` | pickle | BM25 index over the same chunks |

The vector store is a build artefact and is git-ignored; `python -m knowledge.ingest` rebuilds it
deterministically in a few seconds.

**Threshold band schema**

```json
{
  "class": "critically_low",
  "min": 0.0,
  "max": 0.5,
  "interpretation": "Severely depleted; aggregate stability, water holding capacity and microbial habitat all compromised."
}
```

**Claim schema**

```yaml
- id: af_biodiversity
  statement: "Agroforestry systems host 61% higher associated biodiversity (CI 25-105%) than monoculture systems…"
  metric: species_richness
  direction: increase
  magnitude: "+61% (CI 25-105%) associated biodiversity"
  horizon: long
  sources: [beillouin_2021]
  retrieval_query: "agroforestry systems higher biodiversity levels than monocultures percent"
  strength: high
```

---

## 5. Knowledge layer and data sources

The corpus is 14 documents covering all ten required variables. Each is a factual summary of a named
primary source, written for retrieval, with its quantitative findings taken from that source and a
link to the original in `sources.yaml`.

### Peer-reviewed meta-analyses

| Source | Used for |
|---|---|
| Poeplau & Don (2015), *Agric. Ecosyst. Environ.* 200, 33–41 | Cover crop SOC sequestration rate (0.32 ± 0.08 Mg C/ha/yr), saturation behaviour |
| Jian et al. (2020), *Soil Biol. Biochem.* 143, 107735 | Texture-dependence of SOC response, runoff and erosion co-benefits |
| Beillouin et al. (2021), *Global Change Biology* 27(19), 4697–4710 | Effect sizes for every diversification strategy on biodiversity |
| Beillouin et al. (2023), *Nature Communications* 14, 3700 | SOC loss under land-use change; agroforestry +20% SOC in croplands |
| Albrecht et al. (2020), *Ecology Letters* 23(10), 1488–1498 | Flower strips and hedgerows: pest control, distance decay, strip age |
| Kuyah et al. (2019), *Agron. Sustain. Dev.* 39, 47 | Agroforestry in sub-Saharan Africa: yield, SOC, runoff, soil loss |

### Assessment reports and technical guidance

| Source | Used for |
|---|---|
| IPCC SRCCL (2019) | Dryland/aridity classes, land degradation, response options with co-benefits |
| IPCC AR6 WGII (2022) | Drought and species persistence, connectivity, microclimate buffering |
| IPBES Global Assessment (2019) | Ranked drivers of biodiversity loss, fragmentation, land degradation extent |
| IPBES Pollinators Assessment (2016) | Pesticide sub-lethal effects, forage continuity, nesting substrate |
| FAO & ITPS, *Status of the World's Soil Resources* (2015) | Soil threats, acidification chemistry, soil biodiversity |
| FAO, *Recarbonizing Global Soils* (2021) | Practice families, dryland sequencing, permanence, monitoring |
| FAO, *Guidelines for Soil Description*, 4th ed. (2006) | Soil pH classes and their agronomic meaning |

### Compiled mechanism notes

One document (`mechanism_notes.md`) synthesises the causal links described across the sources above
— SOC to water holding capacity, pH as a gate, connectivity versus area, pollution as a multiplier.
It is labelled `internal_synthesis` in the registry, carries no independent empirical claims, and
every quantified figure still comes from a primary source.

### On data integrity

No dataset, percentage, citation or evaluation result in this repository is fabricated. The
quantified effect sizes were verified against the published literature before being written into
`claims.yaml`. Where evidence is synthesis-level rather than a direct measurement, the claim is
marked `strength: synthesis` and recommendations resting only on such claims are automatically
downgraded in confidence.

---

## 6. RAG pipeline

### Ingest — `python -m knowledge.ingest`

1. **Load** every source declared in `sources.yaml` and its corpus file.
2. **Chunk** section-aware: split on markdown `##` headings, then window long sections at 190 words
   with 40-word overlap. Each chunk is prefixed with its document title and section heading before
   embedding so an isolated chunk remains self-describing.
3. **Embed** with a locally fitted TF-IDF → Truncated SVD model (Latent Semantic Analysis), using
   both word (1–2 gram) and character (3–5 gram) features.
4. **Store** chunks, metadata and vectors in a persistent ChromaDB collection (cosine space).
5. **Index** the same chunks with BM25 and persist alongside.

Current build: **14 sources → 80 chunks**, embedding dimension 79.

### Why LSA rather than a transformer encoder

This is a deliberate trade-off, not a limitation of effort:

- **Offline and deterministic** — no model download, no API key, no GPU. The evaluation produces
  byte-identical results anywhere, which is what makes its numbers meaningful.
- **Well matched to corpus size** — on a focused 80-chunk domain corpus, LSA captures term
  co-occurrence structure effectively, and it is paired with BM25 to cover its weak spot.
- **Swappable** — `EMBEDDING_BACKEND=st` switches to `sentence-transformers/all-MiniLM-L6-v2`
  wherever the download is permitted. No other code changes.

### Retrieval — hybrid with Reciprocal Rank Fusion

Dense and lexical search run independently and are fused with RRF (`k=60`):

```
score(chunk) = Σ  1 / (60 + rank_in_that_list)
```

RRF needs no score calibration, which matters because cosine similarity and BM25 scores are not on a
comparable scale. Dense search finds conceptually related passages that share no vocabulary with the
query; BM25 guarantees that a query naming a specific term — *aluminium toxicity*, *aridity index* —
retrieves the passage containing it. Near-duplicate windows of the same section are collapsed so the
returned set spans distinct evidence.

### Claim-scoped retrieval

The step that makes evidence traceable rather than merely present:

```python
evidence_for_claim(claim["retrieval_query"], claim["sources"], k=1)
```

Retrieval is **restricted to the sources the claim declares**. A claim about cover-crop
sequestration rate can only be supported by Poeplau & Don. If that restricted search returns
nothing, `core/evidence.py` removes the claim, removes its metric impact, and downgrades the
recommendation's confidence — with a note in the `uncertainty` field explaining why.

A regression test (`test_every_claim_retrieves_evidence`) asserts that all 38 claims currently
resolve, so a corpus edit that orphans a claim fails CI rather than silently weakening the output.

---

## 7. The reasoning engine

`core/reasoning.py`. Ten rules, in two categories.

### GATE rules — preconditions that reorder advice

Gates are where the non-obvious recommendations come from. A gate can suppress an action rule
entirely.

| Gate | Fires when | Effect |
|---|---|---|
| `gate_pollution` | pollution ≥ high **and** species richness known | Ranks exposure reduction above habitat creation |
| `gate_acid_soil` | pH ≤ strongly_acidic **and** SOC known | **Suppresses** `action_legume_cover_crop`, surfaces liming |
| `gate_dryland_water` | rainfall ≤ semi_arid **and** ≥ 2 of {SOC, moisture, temperature} | Ranks water capture above carbon building |

### ACTION rules

`action_agroforestry` · `action_connectivity` · `action_legume_cover_crop` ·
`action_habitat_strips` · `action_grazing` · `action_rotation` · `action_monitoring`

### The multi-variable requirement is enforced, not aspirational

Every rule declares `requires_min_metrics = 3`. A rule whose conditions match on only two variables
does not fire. Each recommendation carries:

- `metrics_used` — the variables the rule actually inspected
- `interactions` — the causal chains it relied on, e.g.
  `rainfall → biomass ceiling → achievable carbon input`
- `triggered_by` — the specific site conditions, with their values, that fired it

The evaluation asserts `min_multi_metric: 3` on every scenario. Observed across the current suite:
**mean 3.61 variables per recommendation, minimum 3.**

### Confidence

Confidence starts from the rule's base level and is **downgraded** — never upgraded — by:

- a claim that could not be resolved in the registry;
- a claim whose evidence could not be retrieved;
- any input that was assumed rather than supplied;
- supporting evidence that is synthesis-level rather than meta-analytic.

The reason string is printed with the level, so "HIGH" is always accompanied by *why*.

---

## 8. Conversational intelligence

**Clarification is ranked, not exhaustive.** `core/clarify.py` scores each missing variable by a
base priority plus a *contextual* bonus computed from the current state. If rainfall is already
`semi_arid`, soil moisture gains +3.0 because the dryland gate depends on it; if SOC is low, pH
gains +2.5 because it decides whether legumes are viable at all. The system asks for the variable
that would most change its answer.

Each question ships with `why_it_matters`, so the user learns why the variable is being requested.

**The system answers as soon as it can.** Below three variables it asks and gives no advice. At or
above three it produces the full analysis and may attach *one* follow-up question — it does not
withhold the answer to collect more data. Questions are never repeated, and `go ahead` / `not sure`
skips further questioning immediately.

**Memory.** `core/conversation.py` accumulates observations across turns, and a new value for a
known metric overwrites the old one and is logged as a correction — which is what makes
`Actually the pH is 6.9` revise the diagnosis rather than conflict with it. A terse reply
(`0.4`) is attributed to the metric that was just asked about, guarded by a per-metric plausibility
range so a stray number is never recorded as nonsense.

---

## 9. Output contract

`core/render.py` walks the `Recommendation` object and emits every field, so a response is
structurally incapable of omitting its sources or its uncertainty. Each recommendation renders:

**What to do** · **Why it works** (mechanism) · **Why this site specifically** ·
**Cross-variable reasoning** · **Impacted metrics** table (metric, direction, magnitude, horizon) ·
**Time horizon** · **Confidence** with reason · **Uncertainty** · **Caveats and failure modes** ·
**Evidence** (passage, section, organisation, year, URL) · a consolidated **Sources cited** list.

The identical data is available as JSON via `to_json()`, the in-app expander, or `cli.py --json`.

---

## 10. Local setup

Requires Python 3.11+.

```bash
git clone <repository-url>
cd darukaa-bioai

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python -m knowledge.ingest         # builds the vector store (~5 s)

streamlit run app.py               # http://localhost:8501
```

No API key and no `.env` file are required.

**Other entry points**

```bash
python cli.py                                          # interactive, no browser
python cli.py --input "SOC 0.3%, rainfall low, monoculture wheat, 28 C"
python cli.py --json --input '{"soc": 0.3, "rainfall": 380, "land_use": "wheat"}'
python -m eval.run_eval                                # evaluation suite
pytest -q                                              # unit tests
```

---

## 11. Evaluation and tests

### Scenario evaluation — `python -m eval.run_eval`

Ten scenarios in `eval/scenarios.yaml`, each declaring assertions about classifications, which rules
must fire, which must **not** fire, minimum variables per recommendation, and evidence coverage.

```
Scenarios : 10/10 passed
Assertions: 69/69 passed
Mean variables reasoned over per recommendation: 3.61
Minimum variables observed in any recommendation: 3
```

Coverage includes the challenge's worked example, the acid-soil gate **and its control** (neutral pH
must restore the suppressed recommendation), the pollution gate, connectivity after deforestation,
multi-turn accumulation, mid-conversation correction, JSON input, and the grazing pathway.

The evaluation exits non-zero on failure, so it works as a CI gate.

**What these numbers are.** They are behavioural assertions about the system — which rules fire,
whether gates suppress correctly, whether output is multi-variable and evidence-backed. They are
**not** measurements of ecological outcomes, and no field validation is claimed.

### Unit tests — `pytest -q`

**35 tests, all passing**, covering threshold classification, free-text and JSON extraction, unit
conversion (organic matter → organic carbon at 1.724), implausible-value rejection, gate suppression
and its control, the three-variable floor, retrieval attribution and source restriction, referential
integrity between `claims.yaml` and `sources.yaml`, the drop-unsupported-claim path, memory,
correction, reset, and the full output contract.

### CI

`.github/workflows/ci.yml` builds the knowledge base, runs pytest and runs the evaluation on every
push.

---

## 12. Deployment

**Streamlit Community Cloud** (free):

1. Push the repository to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), create an app pointing at `app.py`.
3. No secrets are needed.

The app calls `python -m knowledge.ingest` logic on first load if the store is absent, and the store
is a deterministic build artefact, so no data needs to be committed.

**Docker**

```bash
docker build -t darukaa-bioai .
docker run -p 8501:8501 darukaa-bioai
```

---

## 13. Honest limitations

Stated plainly, because a reviewer will find them anyway.

- **No field validation.** The evaluation tests system behaviour, not ecological accuracy. Effect
  sizes are global or regional meta-analytic means; site-level outcomes vary widely.
- **Geographic skew.** Field-margin and flower-strip evidence is predominantly from northern
  hemisphere temperate systems. The relevant recommendations carry this as an explicit caveat.
- **The corpus is summaries, not full texts.** Fourteen documents summarising named sources, not the
  full PDFs. This keeps retrieval sharp and the repository small, but means the system cannot answer
  questions that depend on detail outside those summaries. The registry links every original.
- **Rules are hand-authored.** Ten rules cover the common cropland, grazing and degraded-land
  situations. An unusual combination produces no recommendation — by design, the system says so
  rather than improvising.
- **No live geospatial lookup.** Coordinates are not resolved to soil or climate data. The
  structured input path is built to accept such values, and SoilGrids/Open-Meteo would slot in at
  `core/extract.py` without touching the reasoning layer.
- **Species richness is compared to qualitative bands.** Absolute richness is only meaningful
  against a local reference site; the system treats these as relative classes and says so in the
  reference table.

---

## 14. Repository map

```
darukaa-bioai/
├── app.py                          Streamlit demo
├── cli.py                          Command-line interface
├── requirements.txt
├── Dockerfile
├── core/
│   ├── schema.py                   Pydantic objects; the contract between layers
│   ├── extract.py                  Text + JSON → Observations
│   ├── reference.py                Structured data layer: value → class
│   ├── clarify.py                  Ranked clarifying-question policy
│   ├── conversation.py             Multi-turn memory and orchestration
│   ├── reasoning.py                MULTI-METRIC RULE GRAPH (gates + actions)
│   ├── evidence.py                 Claim → retrieval; drops unsupported claims
│   ├── retrieve.py                 Hybrid retrieval (Chroma + BM25, RRF)
│   ├── embeddings.py               Local LSA embeddings; ST backend optional
│   └── render.py                   Markdown + JSON output contract
├── knowledge/
│   ├── ingest.py                   Chunk → embed → Chroma + BM25
│   ├── sources.yaml                14 sources with citation metadata
│   ├── claims.yaml                 38 quantified, attributed claims
│   ├── corpus/                     14 source documents
│   └── reference_data/
│       └── thresholds.json         Classification bands for 10 variables
├── eval/
│   ├── scenarios.yaml              10 scenarios with assertions
│   ├── run_eval.py                 Harness; exits non-zero on failure
│   └── REPORT.md                   Generated report
└── tests/
    └── test_system.py              35 unit tests
```

---

Built for the Darukaa.Earth AI Biodiversity Intelligence Challenge.
