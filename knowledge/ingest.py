"""
Knowledge base ingest pipeline.

    python -m knowledge.ingest

Steps
  1. Read every source declared in sources.yaml and load its corpus file.
  2. Split each document into section-aware chunks (markdown '##' headings),
     further splitting long sections on paragraph boundaries with overlap.
  3. Fit the embedding model on the chunk set and write it to disk.
  4. Write chunks + metadata + vectors into a persistent ChromaDB collection.
  5. Build and persist a BM25 index over the same chunks for hybrid retrieval.

Re-running is idempotent: the collection is dropped and rebuilt.
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.embeddings import build_embedding_model  # noqa: E402

KNOWLEDGE_DIR = ROOT / "knowledge"
CORPUS_DIR = KNOWLEDGE_DIR / "corpus"
SOURCES_FILE = KNOWLEDGE_DIR / "sources.yaml"
STORE_DIR = KNOWLEDGE_DIR / "vectorstore"
EMBEDDING_FILE = STORE_DIR / "embedding_model.pkl"
BM25_FILE = STORE_DIR / "bm25.pkl"
COLLECTION_NAME = "darukaa_biodiversity_kb"

MAX_CHUNK_WORDS = 190
OVERLAP_WORDS = 40


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def split_sections(text: str) -> List[Dict[str, str]]:
    """Split markdown into (heading, body) sections on '##' boundaries."""
    lines = text.splitlines()
    doc_title = ""
    sections: List[Dict[str, str]] = []
    current = {"heading": "Overview", "body": []}

    for line in lines:
        if line.startswith("# ") and not doc_title:
            doc_title = line[2:].strip()
            continue
        if line.startswith("## "):
            if current["body"]:
                sections.append({"heading": current["heading"], "body": "\n".join(current["body"]).strip()})
            current = {"heading": line[3:].strip(), "body": []}
        else:
            current["body"].append(line)

    if current["body"]:
        sections.append({"heading": current["heading"], "body": "\n".join(current["body"]).strip()})

    return [s for s in sections if s["body"]]


def window(words: List[str], size: int, overlap: int) -> List[List[str]]:
    if len(words) <= size:
        return [words]
    out, start = [], 0
    step = max(1, size - overlap)
    while start < len(words):
        out.append(words[start : start + size])
        if start + size >= len(words):
            break
        start += step
    return out


def chunk_document(source: Dict, text: str) -> List[Dict]:
    chunks: List[Dict] = []
    for s_idx, section in enumerate(split_sections(text)):
        words = section["body"].split()
        for w_idx, piece in enumerate(window(words, MAX_CHUNK_WORDS, OVERLAP_WORDS)):
            body = " ".join(piece)
            body = re.sub(r"\s+", " ", body).strip()
            if len(body.split()) < 12:
                continue
            chunk_id = f"{source['id']}::s{s_idx}::c{w_idx}"
            # Prepend heading context so an isolated chunk stays self-describing.
            embed_text = f"{source['title']} - {section['heading']}. {body}"
            chunks.append(
                {
                    "id": chunk_id,
                    "text": body,
                    "embed_text": embed_text,
                    "metadata": {
                        "source_id": source["id"],
                        "title": source["title"],
                        "organisation": source["organisation"],
                        "year": source.get("year") or 0,
                        "url": source.get("url") or "",
                        "citation": source.get("citation", ""),
                        "section": section["heading"],
                        "type": source.get("type", ""),
                        "topics": ",".join(source.get("topics", [])),
                        "evidence_strength": source.get("evidence_strength", "unknown"),
                    },
                }
            )
    return chunks


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def load_sources() -> List[Dict]:
    with open(SOURCES_FILE, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["sources"]


def build() -> Dict[str, int]:
    import chromadb
    from rank_bm25 import BM25Okapi

    sources = load_sources()
    all_chunks: List[Dict] = []

    for source in sources:
        path = CORPUS_DIR / source["file"]
        if not path.exists():
            raise FileNotFoundError(f"Corpus file missing for source '{source['id']}': {path}")
        text = path.read_text(encoding="utf-8")
        produced = chunk_document(source, text)
        if not produced:
            raise ValueError(f"Source '{source['id']}' produced no chunks.")
        all_chunks.extend(produced)
        print(f"  {source['id']:<26} {len(produced):>3} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    STORE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Embedding model
    print("Fitting embedding model...")
    model = build_embedding_model()
    model.fit([c["embed_text"] for c in all_chunks])
    model.save(EMBEDDING_FILE)
    vectors = model.encode([c["embed_text"] for c in all_chunks])
    print(f"  embedding dim = {model.dim}")

    # 2. Chroma vector store
    print("Writing Chroma collection...")
    client = chromadb.PersistentClient(path=str(STORE_DIR / "chroma"))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
        embeddings=[v.tolist() for v in vectors],
    )

    # 3. BM25 lexical index
    print("Building BM25 index...")
    tokenised = [tokenise(c["embed_text"]) for c in all_chunks]
    bm25 = BM25Okapi(tokenised)
    with open(BM25_FILE, "wb") as fh:
        pickle.dump(
            {
                "bm25": bm25,
                "ids": [c["id"] for c in all_chunks],
                "texts": [c["text"] for c in all_chunks],
                "metadatas": [c["metadata"] for c in all_chunks],
            },
            fh,
        )

    return {"sources": len(sources), "chunks": len(all_chunks), "dim": model.dim}


def tokenise(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


if __name__ == "__main__":
    print("Building Darukaa.Earth knowledge base\n" + "=" * 42)
    stats = build()
    print("\n" + "=" * 42)
    print(f"Done. {stats['sources']} sources, {stats['chunks']} chunks, dim {stats['dim']}.")
    print(f"Vector store: {STORE_DIR / 'chroma'}")
