from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .chunking import Chunk


class EmbeddingRetriever:
    def __init__(
        self,
        chunks: List[Chunk],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)

        texts = [c.text for c in chunks]
        emb = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embeddings = emb.astype(np.float32)

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        q = (
            self.model.encode(
                [query],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
            .astype(np.float32)
            .reshape(-1)
        )

        sims = (self.embeddings @ q).flatten()
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
