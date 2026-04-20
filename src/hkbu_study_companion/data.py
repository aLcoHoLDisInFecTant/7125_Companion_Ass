"""Data Loading Module

This module is responsible for loading and processing document data, including:
1. Defining the `Doc` data class for storing document information.
2. Loading documents from JSON files.
3. Providing sample document data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import json
from pathlib import Path


@dataclass(frozen=True)
class Doc:
    """Document Data Class
    
    Used to store basic document information, including document ID, title, and text content.
    
    Attributes:
        doc_id: Unique identifier for the document.
        title: Title of the document.
        text: Text content of the document.
    """
    doc_id: str
    title: str
    text: str


def load_docs_from_json(path: str) -> List[Doc]:
    """Loads a list of documents from a JSON file.
    
    Supports three JSON formats:
    1. A list of document objects: [{"doc_id": "...", "title": "...", "text": "..."}, ...]
    2. An object containing a `docs` field: {"docs": [{"doc_id": "...", "title": "...", "text": "..."}, ...]}
    3. JSONL format: one JSON object per line.
    
    Args:
        path: Path to the JSON file.
    
    Returns:
        A list of loaded documents.
    
    Raises:
        ValueError: If the JSON format is unsupported.
    """
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
        # Attempt to process as JSONL format
        lines = [ln for ln in raw_text.splitlines() if ln.strip()]
        items = [json.loads(ln) for ln in lines]

    if items is None:
        raise ValueError("Unsupported JSON format. Use a list of docs or {'docs': [...]} or JSONL.")

    out: List[Doc] = []
    for i, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            continue
        # Generate document ID
        base_id = str(it.get("doc_id") or f"DOC{i}")
        category = it.get("category")
        if isinstance(category, str) and category.strip():
            cat = category.strip()
            doc_id = f"{base_id}:{cat}"
        else:
            doc_id = base_id

        # Generate document title
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

        # Extract document text
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
    """Loads sample documents from Hong Kong Baptist University (HKBU).
    
    Returns a list containing three sample documents:
    1. COMP4146 Course Outline (Summary)
    2. Late Submission Policy
    3. Academic Integrity & Minimum Implementation Requirements
    
    Returns:
        A list of sample documents.
    """
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

