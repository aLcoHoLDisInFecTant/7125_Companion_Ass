from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Optional

from ..data import load_docs_from_json, load_hkbu_sample_docs
from ..pipeline import StudyCompanion


def _tok_total(case: Dict[str, Any]) -> int:
    stats = case.get("token_stats") or {}
    p = int(stats.get("prompt_eval_count") or 0)
    e = int(stats.get("eval_count") or 0)
    return p + e


def _quality_proxy(case: Dict[str, Any]) -> Dict[str, Any]:
    answer = str(case.get("answer") or "").strip()
    citation_count = len(re.findall(r"\[C\d+\]", answer))
    answer_len = len(answer)
    lower = answer.lower()
    uncertainty_markers = [
        "uncertain",
        "not found",
        "insufficient",
        "not enough context",
        "unknown",
    ]
    uncertainty_flag = any(m in lower for m in uncertainty_markers)
    return {
        "answer_len": answer_len,
        "citation_count": citation_count,
        "uncertainty_flag": uncertainty_flag,
    }


def _print_token_comparison(cases: Dict[str, Dict[str, Any]]) -> None:
    if len(cases) < 2:
        return
    print("=" * 90)
    print("[Token Usage Comparison]")
    for name, result in cases.items():
        print(f"- {name}: total_tokens={_tok_total(result)}")
    base = cases.get("Baseline (No RAG)")
    rag = cases.get("Lexical RAG (TF-IDF)")
    if base and rag:
        delta = _tok_total(rag) - _tok_total(base)
        ratio = (_tok_total(rag) / max(1, _tok_total(base))) if _tok_total(base) else 0.0
        print(f"- TF-IDF vs Baseline: delta={delta:+d} tokens, ratio={ratio:.3f}x")


def _print_quality_comparison(cases: Dict[str, Dict[str, Any]]) -> None:
    if len(cases) < 2:
        return
    print("=" * 90)
    print("[Quality Proxy Comparison]")
    for name, result in cases.items():
        qp = _quality_proxy(result)
        print(
            f"- {name}: answer_len={qp['answer_len']}, "
            f"citations={qp['citation_count']}, uncertainty={qp['uncertainty_flag']}"
        )


def _print_case(name: str, result: Dict[str, Any]) -> None:
    print("=" * 90)
    print(f"[{name}]")
    print("- Token stats:", result.get("token_stats"))
    print("- Quality proxy:", _quality_proxy(result))
    print("- Answer:\n", result.get("answer", ""))
    print("\n- Retrieved Context:")
    retrieved: List[Dict[str, Any]] = result.get("retrieved") or []
    if not retrieved:
        print("(None)")
        return
    for i, r in enumerate(retrieved, start=1):
        snippet = str(r.get("text", "")).strip().replace("\n", " ")
        if len(snippet) > 260:
            snippet = snippet[:260].rstrip() + "..."
        score = r.get("score")
        score_str = f"{float(score):.4f}" if score is not None else "NA"
        print(
            f"  [C{i}] score={score_str} | {r.get('doc_id')} | {r.get('chunk_id')} | {r.get('title')}"
        )
        print(f"      {snippet}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3:9b")
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument("--query", default="What is the late submission policy for COMP4146?")
    p.add_argument("--mode", choices=["all", "baseline", "tfidf", "embed"], default="all")
    p.add_argument("--docs-json", default=None)
    p.add_argument("--queries-json", default=None)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--memory-turns", type=int, default=4)
    p.add_argument("--max-ctx-chars", type=int, default=420)
    p.add_argument("--chunk-size", type=int, default=220)
    p.add_argument("--chunk-overlap", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--num-predict", type=int, default=220)
    p.add_argument("--safety-mode", action="store_true")
    p.add_argument("--reasoning-nudge", action="store_true")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args(argv)

    docs = load_docs_from_json(args.docs_json) if args.docs_json else load_hkbu_sample_docs()

    def make_companion() -> StudyCompanion:
        return StudyCompanion(
            docs=docs,
            model=args.model,
            base_url=args.base_url,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            memory_turns=args.memory_turns,
            max_ctx_chars=args.max_ctx_chars,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
            safety_mode=args.safety_mode,
            reasoning_nudge=args.reasoning_nudge,
        )

    queries: List[str] = [args.query]
    if args.queries_json:
        with open(args.queries_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list) or not raw:
            raise ValueError("--queries-json must be a non-empty JSON list of query strings.")
        queries = [str(x).strip() for x in raw if str(x).strip()]
        if not queries:
            raise ValueError("--queries-json provided but no valid query strings found.")

    for idx, query in enumerate(queries, start=1):
        print("#" * 90)
        print(f"[Query {idx}] {query}")
        cases: Dict[str, Dict[str, Any]] = {}

        if args.mode in ("all", "baseline"):
            baseline = make_companion().answer_baseline(query)
            name = "Baseline (No RAG)"
            cases[name] = baseline
            _print_case(name, baseline)

        if args.mode in ("all", "tfidf"):
            lexical = make_companion().answer_tfidf(query, top_k=args.top_k)
            name = "Lexical RAG (TF-IDF)"
            cases[name] = lexical
            _print_case(name, lexical)

        if args.mode in ("all", "embed"):
            neu_companion = make_companion()
            try:
                neural = neu_companion.answer_embed(query, top_k=args.top_k, embed_model=args.embed_model)
                name = "Neural RAG (Embedding)"
                cases[name] = neural
                _print_case(name, neural)
            except Exception as e:
                print("=" * 90)
                print("[Neural RAG (Embedding) FAILED]")
                print(str(e))
                print(
                    "If you are on Windows and see symlink-related errors, enable Developer Mode or run with symlink support, "
                    "or pre-download the embedding model and point --embed-model to a local path."
                )

        _print_token_comparison(cases)
        _print_quality_comparison(cases)


if __name__ == "__main__":
    main()
