"""
Evidence attachment.

The reasoning engine emits claim_ids. This module turns each claim into a
retrieval against the knowledge base, restricted to the sources the claim
declares, and attaches the returned passages to the recommendation.

The important property is what happens on failure: a claim whose supporting
passage cannot be retrieved is *removed*, its metric impact is removed with it,
and the recommendation's confidence is downgraded. The system therefore cannot
present a quantified effect that is not backed by an indexed passage.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from core.reasoning import load_claims
from core.retrieve import evidence_for_claim
from core.schema import Confidence, EvidenceItem, Recommendation

CONFIDENCE_ORDER = [Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH]


def _downgrade(c: Confidence, steps: int = 1) -> Confidence:
    return CONFIDENCE_ORDER[max(0, CONFIDENCE_ORDER.index(c) - steps)]


def attach_evidence(recommendations: List[Recommendation], per_claim: int = 1) -> List[Recommendation]:
    claims = load_claims()
    cache: Dict[str, List[EvidenceItem]] = {}

    for rec in recommendations:
        collected: List[EvidenceItem] = []
        supported: List[str] = []
        unsupported: List[str] = []

        for cid in rec.claim_ids:
            claim = claims.get(cid)
            if claim is None:
                unsupported.append(cid)
                continue
            if cid not in cache:
                cache[cid] = evidence_for_claim(
                    claim["retrieval_query"], claim.get("sources", []), k=per_claim
                )
            items = cache[cid]
            if not items:
                unsupported.append(cid)
                continue
            supported.append(cid)
            for item in items:
                if not any(e.chunk_id == item.chunk_id for e in collected):
                    collected.append(item)

        rec.claim_ids = supported
        rec.evidence = collected
        # Drop impacts whose claim lost its evidence.
        rec.impacts = [i for i in rec.impacts if i.basis is None or i.basis in supported]

        if unsupported:
            rec.confidence = _downgrade(rec.confidence)
            rec.uncertainty.append(
                f"{len(unsupported)} claim(s) had no retrievable supporting passage and were dropped from this recommendation."
            )
        if not supported:
            rec.confidence = Confidence.LOW
            rec.uncertainty.append(
                "No supporting evidence was retrieved; treat this as a hypothesis to verify, not a grounded recommendation."
            )

    return recommendations


def evidence_summary(recommendations: List[Recommendation]) -> Tuple[int, int]:
    """(distinct sources cited, total passages attached) across all recommendations."""
    sources, passages = set(), 0
    for rec in recommendations:
        for e in rec.evidence:
            sources.add(e.source_id)
            passages += 1
    return len(sources), passages
