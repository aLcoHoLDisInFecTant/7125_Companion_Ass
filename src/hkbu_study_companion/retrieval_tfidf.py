from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk


class TfidfRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 2),
            max_features=5000,
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        idx = np.argsort(-sims)[:top_k]
        out: List[Dict[str, Any]] = []
        for rank, i in enumerate(idx, start=1):
            c = self.chunks[int(i)]
            out.append(
                {
                    "rank": rank,
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "score": float(sims[int(i)]),
                    "text": c.text,
                }
            )
        return out
