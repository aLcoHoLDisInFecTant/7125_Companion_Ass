"""Document Chunking Module

This module is responsible for splitting documents into smaller chunks to facilitate subsequent retrieval and processing.
Main features include:
1. Defining the `Chunk` data class for storing chunk information.
2. Providing text chunking functionality.
3. Providing functionality to build a list of chunks from a list of documents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .data import Doc


@dataclass(frozen=True)
class Chunk:
    """Chunk Data Class
    
    Used to store information for a document chunk, including chunk ID, document ID, title, and text content.
    
    Attributes:
        chunk_id: Unique identifier for the chunk.
        doc_id: Unique identifier for the parent document.
        title: Title of the chunk (usually the same as the document title).
        text: Text content of the chunk.
    """
    chunk_id: str
    doc_id: str
    title: str
    text: str


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Splits text into chunks of a specified size.
    
    Args:
        text: The text to be split.
        chunk_size: The size of each chunk (in characters).
        overlap: The overlap size between chunks (in characters).
    
    Returns:
        A list of split text chunks.
    
    Raises:
        ValueError: If `chunk_size` <= 0.
        ValueError: If `overlap` < 0.
        ValueError: If `overlap` >= `chunk_size`.
    """
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks(docs: List[Doc], chunk_size: int, overlap: int) -> List[Chunk]:
    """Builds a list of chunks from a list of documents.
    
    Args:
        docs: List of documents.
        chunk_size: The size of each chunk (in characters).
        overlap: The overlap size between chunks (in characters).
    
    Returns:
        A list of constructed chunks.
    """
    out: List[Chunk] = []
    for d in docs:
        parts = chunk_text(d.text, chunk_size=chunk_size, overlap=overlap)
        for i, p in enumerate(parts, start=1):
            out.append(
                Chunk(
                    chunk_id=f"{d.doc_id}_C{i}",
                    doc_id=d.doc_id,
                    title=d.title,
                    text=p,
                )
            )
    return out

