"""
Local embedding model for the knowledge base.

Design note
-----------
The system uses TF-IDF (word + character n-grams) reduced by Truncated SVD to a
dense vector, i.e. Latent Semantic Analysis. This is a real dense-vector
embedding: semantically related passages that share no exact vocabulary still
land near each other because SVD factors the term co-occurrence structure of the
corpus.

It is chosen over a transformer sentence encoder deliberately:
  * it is fully offline and deterministic, so retrieval is reproducible and the
    evaluation harness gives identical results on any machine;
  * it needs no model download, no API key and no GPU, so the demo deploys
    anywhere;
  * on a focused domain corpus of this size it performs competitively, and
    it is combined with BM25 in a hybrid retriever which covers its weak spot
    (exact rare-term matching).

`EMBEDDING_BACKEND=st` swaps in sentence-transformers/all-MiniLM-L6-v2 when the
environment allows the download; the rest of the system is unchanged.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer

DEFAULT_DIM = 192


class LSAEmbedding:
    """Fit-once, reusable dense embedding model."""

    def __init__(self, dim: int = DEFAULT_DIM):
        self.dim = dim
        self.word_vec: Optional[TfidfVectorizer] = None
        self.char_vec: Optional[TfidfVectorizer] = None
        self.union: Optional[FeatureUnion] = None
        self.svd: Optional[TruncatedSVD] = None
        self.normalizer = Normalizer(copy=False)
        self.fitted = False

    # -- training ---------------------------------------------------------
    def fit(self, texts: List[str]) -> "LSAEmbedding":
        if not texts:
            raise ValueError("Cannot fit an embedding model on an empty corpus.")

        self.word_vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.92,
        )
        # Character n-grams make retrieval robust to morphology and to the
        # scientific vocabulary users misspell ("agroforestry"/"agro forestry").
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=2,
        )
        self.union = FeatureUnion(
            [("word", self.word_vec), ("char", self.char_vec)],
            transformer_weights={"word": 1.0, "char": 0.5},
        )
        matrix = self.union.fit_transform(texts)

        n_components = int(min(self.dim, max(2, min(matrix.shape) - 1)))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = self.svd.fit_transform(matrix)
        self.normalizer.fit(reduced)
        self.dim = n_components
        self.fitted = True
        return self

    # -- inference --------------------------------------------------------
    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Embedding model is not fitted. Run the ingest script first.")
        matrix = self.union.transform(texts)
        reduced = self.svd.transform(matrix)
        # L2 normalise so cosine distance behaves and Chroma's distance is comparable
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (reduced / norms).astype(np.float32)

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0].tolist()

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "LSAEmbedding":
        with open(path, "rb") as fh:
            return pickle.load(fh)


class SentenceTransformerEmbedding:
    """Optional drop-in alternative when network access to model hubs exists."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: F401

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.fitted = True

    def fit(self, texts: List[str]):
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0].tolist()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"backend": "st"}, fh)


def build_embedding_model():
    backend = os.getenv("EMBEDDING_BACKEND", "lsa").lower()
    if backend == "st":
        try:
            return SentenceTransformerEmbedding()
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[embeddings] sentence-transformers unavailable ({exc}); using LSA.")
    return LSAEmbedding()
