"""嵌入检索模块

该模块实现了基于句子嵌入的文本检索功能，使用预训练的句子Transformer模型将文本转换为向量，然后通过计算向量相似度来查找相关内容。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .chunking import Chunk


class EmbeddingRetriever:
    """嵌入检索器
    
    使用预训练的句子Transformer模型将文本转换为向量，通过计算查询向量与文档分块向量之间的相似度来排序结果。
    
    Attributes:
        chunks: 文档分块列表
        device: 运行设备（"cuda"或"cpu"）
        model: 句子Transformer模型
        embeddings: 文档分块的嵌入向量矩阵
    """
    def __init__(
        self,
        chunks: List[Chunk],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """初始化嵌入检索器
        
        Args:
            chunks: 文档分块列表
            model_name: 预训练模型名称，默认为"sentence-transformers/all-MiniLM-L6-v2"
            device: 运行设备，默认为自动检测（优先使用cuda）
        """
        import torch
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        # 自动检测设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # 加载预训练模型
        self.model = SentenceTransformer(model_name, device=self.device)

        # 计算所有文档分块的嵌入向量
        texts = [c.text for c in chunks]
        emb = self.model.encode(
            texts,
            batch_size=64,  # 批处理大小
            show_progress_bar=True,  # 显示进度条
            convert_to_numpy=True,  # 转换为numpy数组
            normalize_embeddings=True,  # 归一化嵌入向量
        )
        self.embeddings = emb.astype(np.float32)  # 转换为float32以节省内存

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """根据查询检索相关文档分块
        
        Args:
            query: 查询文本
            top_k: 返回的top-k结果数
        
        Returns:
            按相关性排序的文档分块列表，每个元素包含rank、chunk_id、doc_id、title、score和text字段
        """
        if top_k <= 0:
            return []
        # 计算查询的嵌入向量
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

        # 计算查询向量与所有文档分块向量的点积（因为向量已归一化，点积等于余弦相似度）
        sims = (self.embeddings @ q).flatten()
        # 获取相似度最高的top-k个分块的索引
        idx = np.argsort(-sims)[:top_k]

        # 构建结果列表
        out: List[Dict[str, Any]] = []
        for rank, i in enumerate(idx, start=1):
            c = self.chunks[int(i)]
            out.append(
                {
                    "rank": rank,  # 排名
                    "chunk_id": c.chunk_id,  # 分块ID
                    "doc_id": c.doc_id,  # 文档ID
                    "title": c.title,  # 标题
                    "score": float(sims[int(i)]),  # 相似度分数
                    "text": c.text,  # 文本内容
                }
            )
        return out
