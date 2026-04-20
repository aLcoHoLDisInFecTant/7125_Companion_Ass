"""Embedding Retrieval Module

This module implements text retrieval functionality based on sentence embeddings. It uses a pre-trained Sentence Transformer model to convert text into vectors and then finds relevant content by calculating vector similarity.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .chunking import Chunk


class EmbeddingRetriever:
    """Embedding Retriever
    
    Uses a pre-trained Sentence Transformer model to convert text into vectors, and sorts results by calculating the similarity between query vectors and document chunk vectors.
    
    Attributes:
        chunks: List of document chunks.
        device: Running device ("cuda" or "cpu").
        model: Sentence Transformer model.
        embeddings: Embedding vector matrix of document chunks.
    """
    def __init__(
        self,
        chunks: List[Chunk],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """Initializes the Embedding Retriever.
        
        Args:
            chunks: List of document chunks.
            model_name: Pre-trained model name, defaults to "sentence-transformers/all-MiniLM-L6-v2".
            device: Running device, defaults to automatic detection (prioritizes cuda).
        """
        import torch
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # Load pre-trained model
        self.model = SentenceTransformer(model_name, device=self.device)

        # Compute embedding vectors for all document chunks
        texts = [c.text for c in chunks]
        emb = self.model.encode(
            texts,
            batch_size=64,  # Batch size
            show_progress_bar=True,  # Show progress bar
            convert_to_numpy=True,  # Convert to numpy array
            normalize_embeddings=True,  # Normalize embedding vectors
        )
        self.embeddings = emb.astype(np.float32)  # Convert to float32 to save memory

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieves relevant document chunks based on the query.
        
        Args:
            query: Query text.
            top_k: Number of top-k results to return.
        
        Returns:
            A list of document chunks sorted by relevance, each containing rank, chunk_id, doc_id, title, score, and text fields.
        """
        if top_k <= 0:
            return []
        # Compute embedding vector for the query
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

        # Calculate the dot product between the query vector and all chunk vectors (since vectors are normalized, dot product equals cosine similarity)
        sims = (self.embeddings @ q).flatten()
        # Get the indices of the top-k chunks with the highest similarity
        idx = np.argsort(-sims)[:top_k]

        # Build the results list
        out: List[Dict[str, Any]] = []
        for rank, i in enumerate(idx, start=1):
            c = self.chunks[int(i)]
            out.append(
                {
                    "rank": rank,  # Rank
                    "chunk_id": c.chunk_id,  # Chunk ID
                    "doc_id": c.doc_id,  # Document ID
                    "title": c.title,  # Title
                    "score": float(sims[int(i)]),  # Similarity score
                    "text": c.text,  # Text content
                }
            )
        return out

