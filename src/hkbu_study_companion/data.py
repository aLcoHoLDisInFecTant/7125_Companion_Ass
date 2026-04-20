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


def _sanitize_doc_id(raw: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in raw)
    out = out.strip("_")
    return out or "DOC"


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


def _load_doc_from_plain_text(path: str) -> List[Doc]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    doc_id = _sanitize_doc_id(p.stem)
    return [Doc(doc_id=doc_id, title=p.name, text=text)]


def _load_doc_from_pdf(path: str) -> List[Doc]:
    p = Path(path)
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError(
            "PDF support requires package 'pypdf'. Please install dependencies again."
        ) from e

    reader = PdfReader(str(p))
    parts: List[str] = []
    for pg in reader.pages:
        try:
            t = pg.extract_text() or ""
        except Exception:
            t = ""
        t = t.strip()
        if t:
            parts.append(t)
    text = "\n\n".join(parts).strip()
    if not text:
        return []
    doc_id = _sanitize_doc_id(p.stem)
    return [Doc(doc_id=doc_id, title=p.name, text=text)]


def _load_doc_from_docx(path: str) -> List[Doc]:
    p = Path(path)
    try:
        from docx import Document
    except Exception as e:
        raise RuntimeError(
            "DOCX support requires package 'python-docx'. Please install dependencies again."
        ) from e

    doc = Document(str(p))
    parts: List[str] = []
    for para in doc.paragraphs:
        t = (para.text or "").strip()
        if t:
            parts.append(t)
    text = "\n".join(parts).strip()
    if not text:
        return []
    doc_id = _sanitize_doc_id(p.stem)
    return [Doc(doc_id=doc_id, title=p.name, text=text)]


def load_docs(path: str) -> List[Doc]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".json", ".jsonl"):
        return load_docs_from_json(path)
    if ext in (".txt", ".md"):
        return _load_doc_from_plain_text(path)
    if ext == ".pdf":
        return _load_doc_from_pdf(path)
    if ext == ".docx":
        return _load_doc_from_docx(path)
    if ext == ".doc":
        raise ValueError(
            "Legacy .doc is not supported directly. Please convert to .docx and retry."
        )
    raise ValueError(
        "Unsupported file format. Supported: .json, .jsonl, .txt, .md, .pdf, .docx"
    )


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
