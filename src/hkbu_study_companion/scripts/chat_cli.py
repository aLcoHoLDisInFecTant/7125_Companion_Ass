from __future__ import annotations

import argparse
from typing import List, Optional

from ..pipeline import StudyCompanion


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3:9b")
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument("--retriever", choices=["baseline", "tfidf", "embed"], default="tfidf")
    p.add_argument("--docs-json", default=None)
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

    companion = StudyCompanion(
        docs_json=args.docs_json,
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

    mode = args.retriever

    print("HKBU Study Companion CLI")
    print("Commands: /mode baseline|tfidf|embed , /clear , /exit")

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/exit", "exit", "quit"):
            break
        if q.startswith("/mode"):
            parts = q.split()
            if len(parts) == 2 and parts[1] in ("baseline", "tfidf", "embed"):
                mode = parts[1]
                print(f"mode = {mode}")
            else:
                print("usage: /mode baseline|tfidf|embed")
            continue
        if q == "/clear":
            companion.memory.turns = []
            print("memory cleared")
            continue

        if mode == "baseline":
            res = companion.answer_baseline(q)
        elif mode == "tfidf":
            res = companion.answer_tfidf(q, top_k=args.top_k)
        else:
            try:
                res = companion.answer_embed(q, top_k=args.top_k, embed_model=args.embed_model)
            except Exception as e:
                print(str(e))
                print(
                    "Embedding retriever failed. If you are on Windows, enable Developer Mode / symlink support, "
                    "or use /mode tfidf for lexical retrieval."
                )
                continue

        print("\nToken stats:", res.get("token_stats"))
        print(res.get("answer", ""))


if __name__ == "__main__":
    main()
