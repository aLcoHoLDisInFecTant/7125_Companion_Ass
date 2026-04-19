"""评估脚本

该脚本用于评估不同检索方法的性能，包括基线（无RAG）、TF-IDF检索和嵌入检索。
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from ..data import load_docs_from_json, load_hkbu_sample_docs
from ..pipeline import StudyCompanion


def _print_case(name: str, result: Dict[str, Any]) -> None:
    """打印评估结果
    
    Args:
        name: 评估方法名称
        result: 评估结果字典
    """
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
    """主函数
    
    Args:
        argv: 命令行参数
    """
    # 解析命令行参数
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3:9b", help="模型名称")
    p.add_argument("--base-url", default="http://localhost:11434", help="Ollama API的基础URL")
    p.add_argument("--query", default="COMP4146 的迟交政策是什么？", help="测试查询")
    p.add_argument("--mode", choices=["all", "baseline", "tfidf", "embed"], default="all", help="评估模式")
    p.add_argument("--docs-json", default=None, help="JSON文档文件路径")
    p.add_argument("--top-k", type=int, default=4, help="检索时返回的top-k结果数")
    p.add_argument("--chunk-size", type=int, default=220, help="文档分块的大小")
    p.add_argument("--chunk-overlap", type=int, default=50, help="文档分块之间的重叠大小")
    p.add_argument("--temperature", type=float, default=0.2, help="温度参数")
    p.add_argument("--top-p", type=float, default=0.9, help="顶部p参数")
    p.add_argument("--num-predict", type=int, default=220, help="预测的token数量")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2", help="嵌入模型名称")
    args = p.parse_args(argv)

    # 加载文档
    docs = load_docs_from_json(args.docs_json) if args.docs_json else load_hkbu_sample_docs()

    # 创建StudyCompanion实例的函数
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

    # 运行基线评估（无RAG）
    if args.mode in ("all", "baseline"):
        baseline = make_companion().answer_baseline(args.query)
        _print_case("Baseline (No RAG)", baseline)

    # 运行TF-IDF检索评估
    if args.mode in ("all", "tfidf"):
        lexical = make_companion().answer_tfidf(args.query, top_k=args.top_k)
        _print_case("Lexical RAG (TF-IDF)", lexical)

    # 运行嵌入检索评估
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
