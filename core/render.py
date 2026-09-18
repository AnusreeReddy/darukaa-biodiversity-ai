"""
Structured output rendering.

The challenge requires every response to state the recommendation, the impacted
metrics, a time horizon, a confidence level and the evidence. That contract is
enforced here rather than left to a language model's discretion: the renderer
walks the Recommendation object and emits every field, so a response is
structurally incapable of omitting its sources or its uncertainty.

Two renderers share one data source: `to_markdown` for humans and `to_json` for
programmatic consumers.
"""

from __future__ import annotations

import json
from typing import Dict, List

from core.schema import Recommendation, SystemResponse

HORIZON_LABEL = {
    "short": "Short term (0-2 years)",
    "medium": "Medium term (2-5 years)",
    "long": "Long term (5+ years)",
}

DIRECTION_SYMBOL = {"increase": "up", "decrease": "down", "stabilise": "stabilise"}


def render_recommendation(rec: Recommendation, index: int = 1) -> str:
    L: List[str] = []
    L.append(f"### {index}. {rec.title}")
    L.append("")
    L.append(f"**What to do** — {rec.action}")
    L.append("")
    L.append(f"**Why it works** — {rec.mechanism}")
    L.append("")

    if rec.triggered_by:
        L.append("**Why this site, specifically**")
        for t in rec.triggered_by:
            L.append(f"- {t}")
        L.append("")

    if rec.interactions:
        L.append(f"**Cross-variable reasoning** (this recommendation combines {rec.n_metrics()} variables: "
                 f"{', '.join(m.value.replace('_', ' ') for m in rec.metrics_used)})")
        for i in rec.interactions:
            L.append(f"- {i}")
        L.append("")

    if rec.impacts:
        L.append("**Impacted metrics**")
        L.append("")
        L.append("| Metric | Direction | Expected magnitude | Horizon |")
        L.append("|---|---|---|---|")
        for imp in rec.impacts:
            L.append(
                f"| {imp.metric.value.replace('_', ' ')} | {DIRECTION_SYMBOL.get(imp.direction.value, imp.direction.value)} "
                f"| {imp.magnitude or 'qualitative'} | {imp.horizon.value} |"
            )
        L.append("")

    L.append(f"**Time horizon** — {HORIZON_LABEL.get(rec.time_horizon.value, rec.time_horizon.value)}")
    L.append("")
    L.append(f"**Confidence** — {rec.confidence.value.upper()} ({rec.confidence_reason})")
    L.append("")

    if rec.uncertainty:
        L.append("**Uncertainty**")
        for u in rec.uncertainty:
            L.append(f"- {u}")
        L.append("")

    if rec.caveats:
        L.append("**Caveats and failure modes**")
        for c in rec.caveats:
            L.append(f"- {c}")
        L.append("")

    if rec.evidence:
        L.append("**Evidence** (retrieved from the indexed knowledge base)")
        for e in rec.evidence:
            ref = f"{e.title} — {e.organisation}"
            if e.year:
                ref += f", {e.year}"
            L.append(f"- *{ref}* — section: {e.section}")
            snippet = e.text[:260].rsplit(" ", 1)[0]
            L.append(f"  > {snippet}...")
            if e.url:
                L.append(f"  [{e.url}]({e.url})")
        L.append("")
    else:
        L.append("**Evidence** — none retrieved; treat as unverified.")
        L.append("")
    return "\n".join(L)


def to_markdown(response: SystemResponse) -> str:
    L: List[str] = []

    if response.message:
        L.append(response.message)
        L.append("")

    if response.interpretation:
        L.append("**Environmental state (interpreted against reference thresholds)**")
        L.append("")
        for line in response.interpretation:
            L.append(f"- {line}")
        L.append("")

    if response.coverage_note:
        L.append(f"*{response.coverage_note}*")
        L.append("")

    if response.mode == "clarify" and response.questions:
        L.append("**To reason properly I need:**")
        L.append("")
        for i, q in enumerate(response.questions, 1):
            L.append(f"{i}. {q.question}")
            L.append(f"   - *Why this matters:* {q.why_it_matters}")
            if q.example_answer:
                L.append(f"   - *{q.example_answer}*")
            L.append("")

    if response.recommendations:
        L.append(f"## Recommendations ({len(response.recommendations)}, ordered by priority)")
        L.append("")
        for i, rec in enumerate(response.recommendations, 1):
            L.append(render_recommendation(rec, i))
            L.append("---")
            L.append("")

        sources = sorted({(e.organisation, e.year, e.title, e.url)
                          for rec in response.recommendations for e in rec.evidence})
        if sources:
            L.append("## Sources cited in this response")
            L.append("")
            for org, year, title, url in sources:
                line = f"- {org}" + (f" ({year})" if year else "") + f". *{title}*."
                if url:
                    line += f" {url}"
                L.append(line)
            L.append("")

    return "\n".join(L)


def to_json(response: SystemResponse) -> Dict:
    """Machine-readable form of the same response."""
    return {
        "mode": response.mode,
        "environmental_state": {
            metric: {
                "value": obs.value,
                "unit": obs.unit,
                "category": obs.category,
                "classification": obs.classification,
                "source": obs.source,
                "turn_recorded": obs.turn,
            }
            for metric, obs in response.state.observations.items()
        },
        "variables_available": len(response.state.known_core_metrics()),
        "coverage_note": response.coverage_note,
        "clarifying_questions": [
            {"metric": q.metric.value, "question": q.question, "why_it_matters": q.why_it_matters}
            for q in response.questions
        ],
        "recommendations": [
            {
                "id": r.intervention_id,
                "title": r.title,
                "action": r.action,
                "mechanism": r.mechanism,
                "triggered_by": r.triggered_by,
                "variables_reasoned_over": [m.value for m in r.metrics_used],
                "variable_count": r.n_metrics(),
                "cross_variable_interactions": r.interactions,
                "impacted_metrics": [
                    {
                        "metric": i.metric.value,
                        "direction": i.direction.value,
                        "magnitude": i.magnitude,
                        "horizon": i.horizon.value,
                        "claim_id": i.basis,
                    }
                    for i in r.impacts
                ],
                "time_horizon": r.time_horizon.value,
                "confidence": r.confidence.value,
                "confidence_reason": r.confidence_reason,
                "uncertainty": r.uncertainty,
                "caveats": r.caveats,
                "claim_ids": r.claim_ids,
                "evidence": [
                    {
                        "chunk_id": e.chunk_id,
                        "source_id": e.source_id,
                        "title": e.title,
                        "organisation": e.organisation,
                        "year": e.year,
                        "section": e.section,
                        "url": e.url,
                        "citation": e.citation,
                        "retrieval_mode": e.retrieval,
                        "passage": e.text,
                    }
                    for e in r.evidence
                ],
            }
            for r in response.recommendations
        ],
    }


def to_json_string(response: SystemResponse) -> str:
    return json.dumps(to_json(response), indent=2, ensure_ascii=False)
