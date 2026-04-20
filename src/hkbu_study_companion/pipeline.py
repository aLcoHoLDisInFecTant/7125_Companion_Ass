from __future__ import annotations

from typing import Any, Dict, List, Optional

from .chunking import Chunk, build_chunks
from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CTX_CHARS,
    DEFAULT_MEMORY_TURNS,
    DEFAULT_MODEL,
    DEFAULT_NUM_PREDICT,
    DEFAULT_REASONING_NUDGE,
    DEFAULT_SAFETY_MODE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from .conversation import ConversationBuffer
from .data import Doc, load_docs_from_json, load_hkbu_sample_docs
from .ollama_client import generate_raw, token_stats
from .prompting import build_prompt, detect_plan_intent, detect_user_language, format_retrieved_context
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
        max_ctx_chars: int = DEFAULT_MAX_CTX_CHARS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        num_predict: int = DEFAULT_NUM_PREDICT,
        safety_mode: bool = DEFAULT_SAFETY_MODE,
        reasoning_nudge: bool = DEFAULT_REASONING_NUDGE,
    ):
        self.model = model
        self.base_url = base_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.memory_turns = memory_turns
        self.max_ctx_chars = max_ctx_chars
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict
        self.safety_mode = safety_mode
        self.reasoning_nudge = reasoning_nudge

        if docs is not None:
            self.docs = docs
        elif docs_json:
            self.docs = load_docs_from_json(docs_json)
        else:
            self.docs = load_hkbu_sample_docs()
        self.chunks: List[Chunk] = build_chunks(
            self.docs, chunk_size=self.chunk_size, overlap=self.chunk_overlap
        )

        self.tfidf = TfidfRetriever(self.chunks)
        self._embed = None

        self.memory = ConversationBuffer(max_turns=self.memory_turns)

    @staticmethod
    def _looks_unsafe(user_query: str) -> bool:
        q = user_query.lower()
        blocked_signals = [
            "cheat",
            "plagiarism",
            "bypass exam",
            "\u4ee3\u5199",
            "\u4f5c\u5f0a",
            "\u67aa\u624b",
            "\u9ed1\u8fdb",
            "hack into",
            "phishing",
            "make a bomb",
            "\u6740\u4eba",
            "\u6bd2\u836f",
        ]
        return any(s in q for s in blocked_signals)

    def ensure_embed(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        if self._embed is not None:
            return
        from .retrieval_embed import EmbeddingRetriever

        self._embed = EmbeddingRetriever(self.chunks, model_name=model_name)

    def _generate(self, user_query: str, retrieved: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        retrieved = retrieved or []
        response_language = detect_user_language(user_query)
        if self.safety_mode and self._looks_unsafe(user_query):
            if response_language == "zh":
                answer = (
                    "\u6211\u4e0d\u80fd\u534f\u52a9\u4f5c\u5f0a\u3001\u8fdd\u6cd5\u6216\u4f24\u5bb3\u4ed6\u4eba\u7684\u8bf7\u6c42\u3002"
                    "\u5982\u679c\u4f60\u662f\u8bfe\u7a0b\u5b66\u4e60\u76f8\u5173\u9700\u6c42\uff0c\u6211\u53ef\u4ee5\u5e2e\u4f60\u6574\u7406\u590d\u4e60\u8ba1\u5212\u3001"
                    "\u89e3\u91ca\u77e5\u8bc6\u70b9\uff0c\u6216\u7ed9\u51fa\u5408\u89c4\u5b66\u4e60\u5efa\u8bae\u3002"
                )
            else:
                answer = (
                    "I cannot help with cheating, illegal activity, or harmful requests. "
                    "If your goal is course learning, I can help with a revision plan, concept explanations, "
                    "or other compliant study guidance."
                )
            self.memory.add_user(user_query)
            self.memory.add_assistant(answer)
            return {
                "answer": answer,
                "retrieved": [],
                "prompt": "(blocked by safety mode before generation)",
                "token_stats": {"prompt_eval_count": 0, "eval_count": 0},
                "raw": {"blocked_by_safety_mode": True},
            }

        has_context = len(retrieved) > 0
        want_plan = detect_plan_intent(user_query)
        prompt = build_prompt(
            user_query=user_query,
            retrieved_context=format_retrieved_context(retrieved, max_chars_each=self.max_ctx_chars),
            conversation_history=self.memory.format_for_prompt(),
            want_plan=want_plan,
            has_context=has_context,
            response_language=response_language,
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
