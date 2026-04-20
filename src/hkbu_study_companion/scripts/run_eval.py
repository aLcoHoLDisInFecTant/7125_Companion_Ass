from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from ..data import load_docs, load_hkbu_sample_docs
from ..pipeline import StudyCompanion


def _print_case(name: str, result: Dict[str, Any]) -> None:
    print("=" * 90)
    print(f"[{name}]")
    print("- Token stats:", result.get("token_stats"))
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
    p.add_argument(
        "--docs-json",
        default=None,
        help="Context file path (.json/.jsonl/.txt/.md/.pdf/.docx)",
    )
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=220)
    p.add_argument("--chunk-overlap", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--num-predict", type=int, default=220)
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args(argv)

    docs = load_docs(args.docs_json) if args.docs_json else load_hkbu_sample_docs()

    def make_companion() -> StudyCompanion:
        return StudyCompanion(
            docs=docs,
            model=args.model,
            base_url=args.base_url,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
        )

    if args.mode in ("all", "baseline"):
        baseline = make_companion().answer_baseline(args.query)
        _print_case("Baseline (No RAG)", baseline)

    if args.mode in ("all", "tfidf"):
        lexical = make_companion().answer_tfidf(args.query, top_k=args.top_k)
        _print_case("Lexical RAG (TF-IDF)", lexical)

    if args.mode in ("all", "embed"):
        neu_companion = make_companion()
        try:
            neural = neu_companion.answer_embed(args.query, top_k=args.top_k, embed_model=args.embed_model)
            _print_case("Neural RAG (Embedding)", neural)
        except Exception as e:
            print("=" * 90)
            print("[Neural RAG (Embedding) FAILED]")
            print(str(e))
            print(
                "If you are on Windows and see symlink-related errors, enable Developer Mode or run with symlink support, "
                "or pre-download the embedding model and point --embed-model to a local path."
            )


if __name__ == "__main__":
    main()
