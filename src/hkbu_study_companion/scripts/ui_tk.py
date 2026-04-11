from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from typing import Optional

import tkinter as tk
from tkinter import filedialog, ttk

from ..pipeline import StudyCompanion


@dataclass
class UiState:
    model: str
    base_url: str
    docs_json: Optional[str]
    retriever: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    temperature: float
    top_p: float
    num_predict: int
    embed_model: str


class App(tk.Tk):
    def __init__(self, state: UiState):
        super().__init__()
        self.state = state
        self.title("HKBU Study Companion")
        self.geometry("1000x700")

        self.companion = StudyCompanion(
            docs_json=self.state.docs_json,
            model=self.state.model,
            base_url=self.state.base_url,
            chunk_size=self.state.chunk_size,
            chunk_overlap=self.state.chunk_overlap,
            top_k=self.state.top_k,
            temperature=self.state.temperature,
            top_p=self.state.top_p,
            num_predict=self.state.num_predict,
        )

        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(root)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Model").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=self.state.model)
        ttk.Entry(top, textvariable=self.model_var, width=18).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(top, text="Retriever").pack(side=tk.LEFT)
        self.retriever_var = tk.StringVar(value=self.state.retriever)
        ttk.Combobox(
            top,
            textvariable=self.retriever_var,
            values=["baseline", "tfidf", "embed"],
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(top, text="Docs").pack(side=tk.LEFT)
        self.docs_var = tk.StringVar(value=self.state.docs_json or "")
        ttk.Entry(top, textvariable=self.docs_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        ttk.Button(top, text="Browse", command=self._browse_docs).pack(side=tk.LEFT)

        mid = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        mid.pack(fill=tk.BOTH, expand=True, pady=(10, 10))

        left = ttk.Frame(mid)
        right = ttk.Frame(mid)
        mid.add(left, weight=2)
        mid.add(right, weight=1)

        ttk.Label(left, text="Chat").pack(anchor="w")
        self.chat = tk.Text(left, wrap=tk.WORD, height=20)
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.configure(state=tk.DISABLED)

        input_row = ttk.Frame(left)
        input_row.pack(fill=tk.X, pady=(8, 0))
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_row, textvariable=self.input_var)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", lambda _e: self._on_send())
        ttk.Button(input_row, text="Send", command=self._on_send).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(right, text="Token stats / Retrieved context").pack(anchor="w")
        self.side = tk.Text(right, wrap=tk.WORD, height=20)
        self.side.pack(fill=tk.BOTH, expand=True)
        self.side.configure(state=tk.DISABLED)

        bottom = ttk.Frame(root)
        bottom.pack(fill=tk.X)

        ttk.Button(bottom, text="Clear", command=self._clear).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Reload Docs", command=self._reload).pack(side=tk.LEFT, padx=(8, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.RIGHT)

        self.input_entry.focus_set()

    def _browse_docs(self) -> None:
        path = filedialog.askopenfilename(
            title="Select JSON/JSONL",
            filetypes=[("JSON/JSONL", "*.json *.jsonl"), ("All files", "*.*")],
        )
        if path:
            self.docs_var.set(path)

    def _append_chat(self, text: str) -> None:
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, text + "\n")
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)

    def _set_side(self, text: str) -> None:
        self.side.configure(state=tk.NORMAL)
        self.side.delete("1.0", tk.END)
        self.side.insert(tk.END, text)
        self.side.see(tk.END)
        self.side.configure(state=tk.DISABLED)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _clear(self) -> None:
        self.companion.memory.turns = []
        self.chat.configure(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.configure(state=tk.DISABLED)
        self._set_side("")
        self._set_status("Ready")

    def _reload(self) -> None:
        docs_json = self.docs_var.get().strip() or None
        self.companion = StudyCompanion(
            docs_json=docs_json,
            model=self.model_var.get().strip() or self.state.model,
            base_url=self.state.base_url,
            chunk_size=self.state.chunk_size,
            chunk_overlap=self.state.chunk_overlap,
            top_k=self.state.top_k,
            temperature=self.state.temperature,
            top_p=self.state.top_p,
            num_predict=self.state.num_predict,
        )
        self._clear()

    def _on_send(self) -> None:
        q = self.input_var.get().strip()
        if not q:
            return
        self.input_var.set("")
        self._append_chat(f"User: {q}")
        self._set_status("Generating...")
        self._set_side("")

        def run() -> None:
            try:
                mode = self.retriever_var.get().strip()
                if mode == "baseline":
                    res = self.companion.answer_baseline(q)
                elif mode == "tfidf":
                    res = self.companion.answer_tfidf(q)
                else:
                    res = self.companion.answer_embed(q, embed_model=self.state.embed_model)
                ans = str(res.get("answer", "")).strip()
                stats = res.get("token_stats")
                retrieved = res.get("retrieved") or []
                side_lines = []
                side_lines.append(f"token_stats: {stats}")
                side_lines.append("")
                side_lines.append("retrieved:")
                if not retrieved:
                    side_lines.append("(None)")
                else:
                    for i, r in enumerate(retrieved, start=1):
                        t = str(r.get("text", "")).strip().replace("\n", " ")
                        if len(t) > 380:
                            t = t[:380].rstrip() + "..."
                        side_lines.append(
                            f"[C{i}] score={r.get('score')} | {r.get('doc_id')} | {r.get('chunk_id')} | {r.get('title')}"
                        )
                        side_lines.append(t)
                        side_lines.append("")
                self.after(0, lambda: self._append_chat(f"Assistant: {ans}\n"))
                self.after(0, lambda: self._set_side("\n".join(side_lines)))
                self.after(0, lambda: self._set_status("Ready"))
            except Exception as e:
                self.after(0, lambda: self._append_chat(f"Assistant: (error) {e}\n"))
                self.after(0, lambda: self._set_status("Ready"))

        threading.Thread(target=run, daemon=True).start()


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma3:4b")
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument("--docs-json", default=None)
    p.add_argument("--retriever", choices=["baseline", "tfidf", "embed"], default="tfidf")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=220)
    p.add_argument("--chunk-overlap", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--num-predict", type=int, default=220)
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args(argv)

    app = App(
        UiState(
            model=args.model,
            base_url=args.base_url,
            docs_json=args.docs_json,
            retriever=args.retriever,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
            embed_model=args.embed_model,
        )
    )
    app.mainloop()


if __name__ == "__main__":
    main()

