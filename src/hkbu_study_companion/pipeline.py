"""Core Pipeline Module

This module is the heart of the system, responsible for integrating various components and providing a unified interface.
Main features include:
1. Initializing the system, loading documents, and building chunks.
2. Providing different response methods (Baseline, TF-IDF Retrieval, Embedding Retrieval).
3. Managing conversation history and generating responses.
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
    """Study Companion Main Class
    
    Integrates document retrieval, conversation management, and text generation to provide a unified interface.
    
    Attributes:
        model: Model name.
        base_url: Base URL of the Ollama API.
        chunk_size: Size of document chunks.
        chunk_overlap: Overlap size between document chunks.
        top_k: Number of top-k results to return during retrieval.
        temperature: Temperature parameter, controlling the randomness of generated text.
        top_p: Top-p parameter, controlling the diversity of generated text.
        num_predict: Number of tokens to predict.
        docs: List of documents.
        chunks: List of document chunks.
        tfidf: TF-IDF retriever.
        _embed: Embedding retriever (lazy-loaded).
        memory: Conversation buffer.
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
        """Initializes the Study Companion.
        
        Args:
            docs: List of documents to use, if provided.
            docs_json: Path to a JSON file to load documents from, if provided.
            model: Model name.
            base_url: Base URL of the Ollama API.
            chunk_size: Size of document chunks.
            chunk_overlap: Overlap size between document chunks.
            top_k: Number of top-k results to return during retrieval.
            memory_turns: Maximum number of conversation turns.
            temperature: Temperature parameter, controlling the randomness of generated text.
            top_p: Top-p parameter, controlling the diversity of generated text.
            num_predict: Number of tokens to predict.
        """
        self.model = model
        self.base_url = base_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict

        # Load documents
        if docs is not None:
            self.docs = docs
        elif docs_json:
            self.docs = load_docs_from_json(docs_json)
        else:
            self.docs = load_hkbu_sample_docs()
        
        # Build document chunks
        self.chunks: List[Chunk] = build_chunks(
            self.docs, chunk_size=self.chunk_size, overlap=self.chunk_overlap
        )

        # Initialize TF-IDF retriever
        self.tfidf = TfidfRetriever(self.chunks)
        # Lazy-load embedding retriever
        self._embed = None

        # Initialize conversation buffer
        self.memory = ConversationBuffer(max_turns=memory_turns)

    def ensure_embed(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Ensures the embedding retriever is initialized.
        
        Args:
            model_name: Pre-trained model name.
        """
        if self._embed is not None:
            return
        from .retrieval_embed import EmbeddingRetriever

        self._embed = EmbeddingRetriever(self.chunks, model_name=model_name)

    def _generate(self, user_query: str, retrieved: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Generates a response.
        
        Args:
            user_query: User query.
            retrieved: List of retrieved document chunks.
        
        Returns:
            A dictionary containing the answer, retrieval results, prompt, token stats, and raw response.
        """
        retrieved = retrieved or []
        has_context = len(retrieved) > 0
        want_plan = detect_plan_intent(user_query)
        
        # Build the prompt template
        prompt = build_prompt(
            user_query=user_query,
            retrieved_context=format_retrieved_context(retrieved),
            conversation_history=self.memory.format_for_prompt(),
            want_plan=want_plan,
            has_context=has_context,
        )
        
        # Call the Ollama API to generate the response
        resp = generate_raw(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            num_predict=self.num_predict,
            stream=False,
            base_url=self.base_url,
        )
        
        # Process the response
        answer = str(resp.get("response", "")).strip()
        # Add to conversation history
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
        """Responds using the baseline method (No RAG).
        
        Args:
            user_query: User query.
        
        Returns:
            A dictionary containing the answer, retrieval results, prompt, token stats, and raw response.
        """
        return self._generate(user_query, retrieved=[])

    def answer_tfidf(self, user_query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Responds using TF-IDF retrieval.
        
        Args:
            user_query: User query.
            top_k: Number of top-k results to return, defaults to the setting at initialization.
        
        Returns:
            A dictionary containing the answer, retrieval results, prompt, token stats, and raw response.
        """
        k = self.top_k if top_k is None else top_k
        retrieved = self.tfidf.retrieve(user_query, top_k=k)
        return self._generate(user_query, retrieved=retrieved)

    def answer_embed(self,
        user_query: str,
        top_k: Optional[int] = None,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> Dict[str, Any]:
        """Responds using embedding retrieval.
        
        Args:
            user_query: User query.
            top_k: Number of top-k results to return, defaults to the setting at initialization.
            embed_model: Pre-trained model name.
        
        Returns:
            A dictionary containing the answer, retrieval results, prompt, token stats, and raw response.
        """
        self.ensure_embed(model_name=embed_model)
        k = self.top_k if top_k is None else top_k
        retrieved = self._embed.retrieve(user_query, top_k=k)
        return self._generate(user_query, retrieved=retrieved)

