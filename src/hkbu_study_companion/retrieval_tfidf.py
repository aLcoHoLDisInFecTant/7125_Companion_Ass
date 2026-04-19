"""TF-IDF检索模块

该模块实现了基于TF-IDF的文本检索功能，用于从文档分块中查找与查询相关的内容。
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk


class TfidfRetriever:
    """TF-IDF检索器
    
    使用TF-IDF（词频-逆文档频率）算法实现文本检索，通过计算查询与文档分块之间的余弦相似度来排序结果。
    
    Attributes:
        chunks: 文档分块列表
        vectorizer: TF-IDF向量化器
        matrix: 文档分块的TF-IDF矩阵
    """
    def __init__(self, chunks: List[Chunk]):
        """初始化TF-IDF检索器
        
        Args:
            chunks: 文档分块列表
        """
        self.chunks = chunks
        # 初始化TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            lowercase=True,  # 转换为小写
            stop_words=None,  # 不使用停用词
            ngram_range=(1, 2),  # 考虑1-gram和2-gram
            max_features=5000,  # 最大特征数
        )
        # 构建TF-IDF矩阵
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

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
        # 计算查询的TF-IDF向量
        q_vec = self.vectorizer.transform([query])
        # 计算查询与所有文档分块的余弦相似度
        sims = cosine_similarity(q_vec, self.matrix).flatten()
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
