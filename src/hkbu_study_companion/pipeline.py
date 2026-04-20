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
from .data import Doc, load_docs, load_hkbu_sample_docs
from .ollama_client import generate_raw, token_stats
from .prompting import build_prompt, detect_plan_intent, format_retrieved_context
from .retrieval_tfidf import TfidfRetriever


class StudyCompanion:
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
        ctx_chars_each: int = 420,
        safety_mode: bool = False,
        reasoning_nudge: bool = False,
    ):
        self.model = model
        self.base_url = base_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict
        self.ctx_chars_each = ctx_chars_each
        self.safety_mode = safety_mode
        self.reasoning_nudge = reasoning_nudge

        if docs is not None:
            self.docs = docs
        elif docs_json:
            self.docs = load_docs(docs_json)
        else:
            self.docs = load_hkbu_sample_docs()
        self.chunks: List[Chunk] = build_chunks(
            self.docs, chunk_size=self.chunk_size, overlap=self.chunk_overlap
        )

        self.tfidf = TfidfRetriever(self.chunks)
        self._embed = None

        self.memory = ConversationBuffer(max_turns=memory_turns)

    def ensure_embed(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        if self._embed is not None:
            return
        from .retrieval_embed import EmbeddingRetriever

        self._embed = EmbeddingRetriever(self.chunks, model_name=model_name)

    def _generate(self, user_query: str, retrieved: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        retrieved = retrieved or []
        has_context = len(retrieved) > 0
        want_plan = detect_plan_intent(user_query)
        prompt = build_prompt(
            user_query=user_query,
            retrieved_context=format_retrieved_context(
                retrieved,
                max_chars_each=self.ctx_chars_each,
            ),
            conversation_history=self.memory.format_for_prompt(),
            want_plan=want_plan,
            has_context=has_context,
            safety_mode=self.safety_mode,
            reasoning_nudge=self.reasoning_nudge,
        )
        resp = generate_raw(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            num_predict=self.num_predict,
            stream=False,
            base_url=self.base_url,
        )
        answer = str(resp.get("response", "")).strip()
        if not answer:
            # Guard against occasional empty model outputs so UI/CLI never shows a blank answer.
            if has_context:
                answer = (
                    "I could not generate a complete answer from the current context. "
                    "Please try again, reduce top_k/ctx_chars, or switch to another model."
                )
            else:
                answer = (
                    "I could not generate a complete answer for this request. "
                    "Please try again or switch to another model."
                )
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
        return self._generate(user_query, retrieved=[])

    def answer_tfidf(self, user_query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        k = self.top_k if top_k is None else top_k
        retrieved = self.tfidf.retrieve(user_query, top_k=k)
        return self._generate(user_query, retrieved=retrieved)

    def answer_embed(
        self,
        user_query: str,
        top_k: Optional[int] = None,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> Dict[str, Any]:
        self.ensure_embed(model_name=embed_model)
        k = self.top_k if top_k is None else top_k
        retrieved = self._embed.retrieve(user_query, top_k=k)
        return self._generate(user_query, retrieved=retrieved)
