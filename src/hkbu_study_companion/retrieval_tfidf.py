"""TF-IDF Retrieval Module

This module implements text retrieval functionality based on TF-IDF, used to find content relevant to a query from document chunks.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk


class TfidfRetriever:
    """TF-IDF Retriever
    
    Implements text retrieval using the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm, sorting results by calculating the cosine similarity between the query and document chunks.
    
    Attributes:
        chunks: List of document chunks.
        vectorizer: TF-IDF vectorizer.
        matrix: TF-IDF matrix of document chunks.
    """
    def __init__(self, chunks: List[Chunk]):
        """Initializes the TF-IDF Retriever.
        
        Args:
            chunks: List of document chunks.
        """
        self.chunks = chunks
        # Initialize the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,  # Convert to lowercase
            stop_words=None,  # Do not use stop words
            ngram_range=(1, 2),  # Consider 1-gram and 2-gram
            max_features=5000,  # Maximum number of features
        )
        # Build the TF-IDF matrix
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

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
        # Calculate the TF-IDF vector for the query
        q_vec = self.vectorizer.transform([query])
        # Calculate cosine similarity between the query and all document chunks
        sims = cosine_similarity(q_vec, self.matrix).flatten()
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

