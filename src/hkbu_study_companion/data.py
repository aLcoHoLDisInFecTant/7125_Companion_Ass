from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import json
from pathlib import Path


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str


def load_docs_from_json(path: str) -> List[Doc]:
    p = Path(path)
    raw_text = p.read_text(encoding="utf-8")
    items: Optional[List[dict]] = None

    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict) and isinstance(obj.get("docs"), list):
            items = obj["docs"]
        elif isinstance(obj, list):
            items = obj
    except json.JSONDecodeError:
        lines = [ln for ln in raw_text.splitlines() if ln.strip()]
        items = [json.loads(ln) for ln in lines]

    if items is None:
        raise ValueError("Unsupported JSON format. Use a list of docs or {'docs': [...]} or JSONL.")

    out: List[Doc] = []
    for i, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            continue
        base_id = str(it.get("doc_id") or f"DOC{i}")
        category = it.get("category")
        if isinstance(category, str) and category.strip():
            cat = category.strip()
            doc_id = f"{base_id}:{cat}"
        else:
            doc_id = base_id

        source_file = it.get("source_file")
        if it.get("title"):
            title = str(it.get("title"))
        else:
            parts = []
            if isinstance(source_file, str) and source_file.strip():
                parts.append(source_file.strip())
            if isinstance(category, str) and category.strip():
                parts.append(category.strip())
            title = " | ".join(parts) if parts else doc_id

        text = str(
            it.get("text")
            or it.get("clean_markdown_content")
            or it.get("content")
            or it.get("summary")
            or ""
        )
        if not text.strip():
            continue
        out.append(Doc(doc_id=doc_id, title=title, text=text))
    return out


def load_hkbu_sample_docs() -> List[Doc]:
    return [
        Doc(
            doc_id="DOC1",
            title="COMP4146 Course Outline (Summary)",
            text=(
                "COMP4146 Data Analytics and Artificial Intelligence is a course focusing on practical and theoretical foundations.\n"
                "Core modules include: data preprocessing, exploratory data analysis, classical machine learning, evaluation metrics,\n"
                "introduction to deep learning, and an applied project component emphasizing responsible AI practices.\n"
                "Assessment typically includes assignments, quizzes, and a final project.\n"
            ),
        ),
        Doc(
            doc_id="DOC2",
            title="Late Submission Policy",
            text=(
                "Late submission policy: zero tolerance without prior approval.\n"
                "If an assignment is submitted after the deadline and no prior approval is granted, the submission is not accepted.\n"
                "Students should request extensions before the deadline with valid reasons and supporting evidence.\n"
            ),
        ),
        Doc(
            doc_id="DOC3",
            title="Academic Integrity & Minimum Implementation Requirements",
            text=(
                "Academic integrity: plagiarism and unauthorized collaboration are prohibited.\n"
                "You must cite sources properly and submit your own work.\n"
                "Minimum implementation requirements checklist may include: reproducible environment, clear documentation,\n"
                "baseline comparison, evaluation metrics, and ablation or error analysis where applicable.\n"
            ),
        ),
    ]
