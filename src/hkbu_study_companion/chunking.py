"""文档分块模块

该模块负责将文档分割成小块，以便于后续的检索和处理。
主要功能包括：
1. 定义Chunk数据类，用于存储分块信息
2. 提供文本分块功能
3. 提供从文档列表构建分块列表的功能
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .data import Doc


@dataclass(frozen=True)
class Chunk:
    """分块数据类
    
    用于存储文档分块的信息，包括分块ID、文档ID、标题和文本内容。
    
    Attributes:
        chunk_id: 分块唯一标识符
        doc_id: 所属文档的唯一标识符
        title: 分块的标题（通常与文档标题相同）
        text: 分块的文本内容
    """
    chunk_id: str
    doc_id: str
    title: str
    text: str


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """将文本分割成指定大小的块
    
    Args:
        text: 要分割的文本
        chunk_size: 每个块的大小（字符数）
        overlap: 块之间的重叠大小（字符数）
    
    Returns:
        分割后的文本块列表
    
    Raises:
        ValueError: 如果chunk_size <= 0
        ValueError: 如果overlap < 0
        ValueError: 如果overlap >= chunk_size
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
    """从文档列表构建分块列表
    
    Args:
        docs: 文档列表
        chunk_size: 每个块的大小（字符数）
        overlap: 块之间的重叠大小（字符数）
    
    Returns:
        构建的分块列表
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
