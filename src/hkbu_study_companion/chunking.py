from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .data import Doc


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
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
