"""核心 pipeline 模块

该模块是系统的核心，负责整合各个组件并提供统一的接口。
主要功能包括：
1. 初始化系统，加载文档和构建分块
2. 提供不同的回答方法（基线、TF-IDF检索、嵌入检索）
3. 管理对话历史和生成回答
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .chunking import Chunk, build_chunks
from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MEMORY_TURNS,
    DEFAULT_MODEL,
    DEFAULT_NUM_PREDICT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from .conversation import ConversationBuffer
from .data import Doc, load_docs_from_json, load_hkbu_sample_docs
from .ollama_client import generate_raw, token_stats
from .prompting import build_prompt, detect_plan_intent, format_retrieved_context
from .retrieval_tfidf import TfidfRetriever


class StudyCompanion:
    """学习助手主类
    
    整合文档检索、对话管理和文本生成功能，提供统一的接口。
    
    Attributes:
        model: 模型名称
        base_url: Ollama API的基础URL
        chunk_size: 文档分块的大小
        chunk_overlap: 文档分块之间的重叠大小
        top_k: 检索时返回的top-k结果数
        temperature: 温度参数，控制生成文本的随机性
        top_p: 顶部p参数，控制生成文本的多样性
        num_predict: 预测的token数量
        docs: 文档列表
        chunks: 文档分块列表
        tfidf: TF-IDF检索器
        _embed: 嵌入检索器（延迟加载）
        memory: 对话缓冲区
    """
    def __init__(
        self,
        *,
        docs: Optional[List[Doc]] = None,
        docs_json: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        memory_turns: int = DEFAULT_MEMORY_TURNS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        num_predict: int = DEFAULT_NUM_PREDICT,
    ):
        """初始化学习助手
        
        Args:
            docs: 文档列表，如果提供则使用
            docs_json: JSON文件路径，如果提供则从文件加载文档
            model: 模型名称
            base_url: Ollama API的基础URL
            chunk_size: 文档分块的大小
            chunk_overlap: 文档分块之间的重叠大小
            top_k: 检索时返回的top-k结果数
            memory_turns: 对话历史的最大轮次数
            temperature: 温度参数，控制生成文本的随机性
            top_p: 顶部p参数，控制生成文本的多样性
            num_predict: 预测的token数量
        """
        self.model = model
        self.base_url = base_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict

        # 加载文档
        if docs is not None:
            self.docs = docs
        elif docs_json:
            self.docs = load_docs_from_json(docs_json)
        else:
            self.docs = load_hkbu_sample_docs()
        
        # 构建文档分块
        self.chunks: List[Chunk] = build_chunks(
            self.docs, chunk_size=self.chunk_size, overlap=self.chunk_overlap
        )

        # 初始化TF-IDF检索器
        self.tfidf = TfidfRetriever(self.chunks)
        # 嵌入检索器延迟加载
        self._embed = None

        # 初始化对话缓冲区
        self.memory = ConversationBuffer(max_turns=memory_turns)

    def ensure_embed(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """确保嵌入检索器已初始化
        
        Args:
            model_name: 预训练模型名称
        """
        if self._embed is not None:
            return
        from .retrieval_embed import EmbeddingRetriever

        self._embed = EmbeddingRetriever(self.chunks, model_name=model_name)

    def _generate(self, user_query: str, retrieved: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """生成回答
        
        Args:
            user_query: 用户查询
            retrieved: 检索到的文档分块列表
        
        Returns:
            包含回答、检索结果、提示模板、token统计和原始响应的字典
        """
        retrieved = retrieved or []
        has_context = len(retrieved) > 0
        want_plan = detect_plan_intent(user_query)
        
        # 构建提示模板
        prompt = build_prompt(
            user_query=user_query,
            retrieved_context=format_retrieved_context(retrieved),
            conversation_history=self.memory.format_for_prompt(),
            want_plan=want_plan,
            has_context=has_context,
        )
        
        # 调用Ollama API生成回答
        resp = generate_raw(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            num_predict=self.num_predict,
            stream=False,
            base_url=self.base_url,
        )
        
        # 处理回答
        answer = str(resp.get("response", "")).strip()
        # 添加到对话历史
        self.memory.add_user(user_query)
        self.memory.add_assistant(answer)
        
        return {
            "answer": answer,
            "retrieved": retrieved,
            "prompt": prompt,
            "token_stats": token_stats(resp),
            "raw": resp,
        }

    def answer_baseline(self, user_query: str) -> Dict[str, Any]:
        """使用基线方法回答（无RAG）
        
        Args:
            user_query: 用户查询
        
        Returns:
            包含回答、检索结果、提示模板、token统计和原始响应的字典
        """
        return self._generate(user_query, retrieved=[])

    def answer_tfidf(self, user_query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """使用TF-IDF检索回答
        
        Args:
            user_query: 用户查询
            top_k: 返回的top-k结果数，默认使用初始化时的设置
        
        Returns:
            包含回答、检索结果、提示模板、token统计和原始响应的字典
        """
        k = self.top_k if top_k is None else top_k
        retrieved = self.tfidf.retrieve(user_query, top_k=k)
        return self._generate(user_query, retrieved=retrieved)

    def answer_embed(self,
        user_query: str,
        top_k: Optional[int] = None,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> Dict[str, Any]:
        """使用嵌入检索回答
        
        Args:
            user_query: 用户查询
            top_k: 返回的top-k结果数，默认使用初始化时的设置
            embed_model: 预训练模型名称
        
        Returns:
            包含回答、检索结果、提示模板、token统计和原始响应的字典
        """
        self.ensure_embed(model_name=embed_model)
        k = self.top_k if top_k is None else top_k
        retrieved = self._embed.retrieve(user_query, top_k=k)
        return self._generate(user_query, retrieved=retrieved)
