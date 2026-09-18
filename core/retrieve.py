"""
Hybrid retrieval over the biodiversity knowledge base.

Dense vector search (Chroma, cosine) and BM25 lexical search are run
independently and fused with Reciprocal Rank Fusion. Dense search finds
passages that are conceptually related but share no vocabulary with the query;
BM25 guarantees that a query naming a specific term ("aluminium toxicity",
"aridity index") retrieves the passage containing it. RRF needs no score
calibration between the two, which matters because cosine similarity and BM25
scores are not on a comparable scale.

Retrieval can be constrained to specific source documents. The reasoning engine
uses this to fetch evidence for a *named claim* rather than for a vague topic,
which is what makes each recommendation traceable to a specific study.
"""

from __future__ import annotations

import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from core.embeddings import LSAEmbedding
from core.schema import EvidenceItem

ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = ROOT / "knowledge" / "vectorstore"
EMBEDDING_FILE = STORE_DIR / "embedding_model.pkl"
BM25_FILE = STORE_DIR / "bm25.pkl"
COLLECTION_NAME = "darukaa_biodiversity_kb"

RRF_K = 60


class KnowledgeBaseMissing(RuntimeError):
    pass


def tokenise(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _load_resources():
    if not EMBEDDING_FILE.exists() or not BM25_FILE.exists():
        raise KnowledgeBaseMissing(
            "Knowledge base not built. Run:  python -m knowledge.ingest"
        )
    import chromadb

    model = LSAEmbedding.load(EMBEDDING_FILE)
    with open(BM25_FILE, "rb") as fh:
        lexical = pickle.load(fh)
    client = chromadb.PersistentClient(path=str(STORE_DIR / "chroma"))
    collection = client.get_collection(COLLECTION_NAME)
    return model, lexical, collection


def kb_is_ready() -> bool:
    try:
        _load_resources()
        return True
    except Exception:
        return False


def kb_stats() -> Dict[str, int]:
    model, lexical, collection = _load_resources()
    return {
        "chunks": collection.count(),
        "sources": len({m["source_id"] for m in lexical["metadatas"]}),
        "dim": model.dim,
    }


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

def _vector_search(query: str, k: int, source_ids: Optional[Sequence[str]]) -> List[str]:
    model, _, collection = _load_resources()
    where = None
    if source_ids:
        where = (
            {"source_id": source_ids[0]}
            if len(source_ids) == 1
            else {"$or": [{"source_id": s} for s in source_ids]}
        )
    res = collection.query(
        query_embeddings=[model.encode_one(query)],
        n_results=k,
        where=where,
    )
    return res["ids"][0] if res["ids"] else []


def _bm25_search(query: str, k: int, source_ids: Optional[Sequence[str]]) -> List[str]:
    _, lexical, _ = _load_resources()
    scores = lexical["bm25"].get_scores(tokenise(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    picked: List[str] = []
    allowed = set(source_ids) if source_ids else None
    for i in order:
        if allowed and lexical["metadatas"][i]["source_id"] not in allowed:
            continue
        if scores[i] <= 0:
            continue
        picked.append(lexical["ids"][i])
        if len(picked) >= k:
            break
    return picked


def _chunk_lookup() -> Dict[str, Dict]:
    _, lexical, _ = _load_resources()
    return {
        cid: {"text": txt, "metadata": md}
        for cid, txt, md in zip(lexical["ids"], lexical["texts"], lexical["metadatas"])
    }


def search(
    query: str,
    k: int = 4,
    source_ids: Optional[Sequence[str]] = None,
    pool: int = 12,
) -> List[EvidenceItem]:
    """Hybrid search returning ranked, fully attributed evidence items."""
    if not query.strip():
        return []

    dense = _vector_search(query, pool, source_ids)
    lexical_hits = _bm25_search(query, pool, source_ids)

    # Reciprocal Rank Fusion
    fused: Dict[str, float] = {}
    modes: Dict[str, set] = {}
    for rank_, cid in enumerate(dense):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank_ + 1)
        modes.setdefault(cid, set()).add("vector")
    for rank_, cid in enumerate(lexical_hits):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank_ + 1)
        modes.setdefault(cid, set()).add("bm25")

    lookup = _chunk_lookup()
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    items: List[EvidenceItem] = []
    seen_sections = set()
    for cid, score in ordered:
        entry = lookup.get(cid)
        if not entry:
            continue
        md = entry["metadata"]
        # Avoid returning several near-identical windows of one section.
        key = (md["source_id"], md["section"])
        if key in seen_sections:
            continue
        seen_sections.add(key)
        items.append(
            EvidenceItem(
                chunk_id=cid,
                text=entry["text"],
                source_id=md["source_id"],
                title=md["title"],
                organisation=md["organisation"],
                year=md["year"] or None,
                url=md["url"] or None,
                citation=md["citation"],
                section=md["section"],
                score=round(score, 5),
                retrieval="+".join(sorted(modes.get(cid, {"hybrid"}))),
            )
        )
        if len(items) >= k:
            break
    return items


def evidence_for_claim(
    query: str, source_ids: Sequence[str], k: int = 2
) -> List[EvidenceItem]:
    """
    Fetch evidence restricted to the sources a claim declares.

    If the restricted search returns nothing, the claim is treated as
    unsupported. The reasoning layer drops or downgrades such claims rather
    than letting the language model fill the gap.
    """
    items = search(query, k=k, source_ids=source_ids, pool=10)
    if not items:
        items = search(query, k=k, source_ids=None, pool=10)
        for it in items:
            it.retrieval += "(fallback)"
    return items
