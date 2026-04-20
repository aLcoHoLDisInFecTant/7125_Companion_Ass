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
    memory_turns: int
    max_ctx_chars: int
    top_k: int
    chunk_size: int
    chunk_overlap: int
    temperature: float
    top_p: float
    num_predict: int
    safety_mode: bool
    reasoning_nudge: bool
    embed_model: str


class App(tk.Tk):
    def __init__(self, state: UiState):
        super().__init__()
        self.state = state
        self.title("HKBU Study Companion")
        self.geometry("1280x820")

        self.companion = StudyCompanion(
            docs_json=self.state.docs_json,
            model=self.state.model,
            base_url=self.state.base_url,
            chunk_size=self.state.chunk_size,
            chunk_overlap=self.state.chunk_overlap,
            top_k=self.state.top_k,
            memory_turns=self.state.memory_turns,
            max_ctx_chars=self.state.max_ctx_chars,
            temperature=self.state.temperature,
            top_p=self.state.top_p,
            num_predict=self.state.num_predict,
            safety_mode=self.state.safety_mode,
            reasoning_nudge=self.state.reasoning_nudge,
        )
        self.is_generating = False

        self._build_ui()
        self._update_memory_status()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        top_row = ttk.Frame(root)
        top_row.pack(fill=tk.X)

        ttk.Label(top_row, text="Model").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=self.state.model)
        ttk.Entry(top_row, textvariable=self.model_var, width=16).pack(side=tk.LEFT, padx=(6, 12))

        ttk.Label(top_row, text="Retriever").pack(side=tk.LEFT)
        self.retriever_var = tk.StringVar(value=self.state.retriever)
        ttk.Combobox(
            top_row,
            textvariable=self.retriever_var,
            values=["baseline", "tfidf", "embed"],
            state="readonly",
            width=9,
        ).pack(side=tk.LEFT, padx=(6, 12))

        ttk.Label(top_row, text="Docs").pack(side=tk.LEFT)
        self.docs_var = tk.StringVar(value=self.state.docs_json or "")
        ttk.Entry(top_row, textvariable=self.docs_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6)
        )
        ttk.Button(top_row, text="Browse", command=self._browse_docs).pack(side=tk.LEFT)

        gen_box = ttk.LabelFrame(root, text="Generation & retrieval (token efficiency / Topic 2)")
        gen_box.pack(fill=tk.X, pady=(10, 0))
        self.temperature_var = tk.StringVar(value=f"{self.state.temperature:.2f}")
        self.top_p_var = tk.StringVar(value=f"{self.state.top_p:.2f}")
        self.num_predict_var = tk.StringVar(value=str(self.state.num_predict))
        self.top_k_var = tk.StringVar(value=str(self.state.top_k))
        self.mem_pairs_var = tk.StringVar(value=str(self.state.memory_turns))
        self.ctx_chars_var = tk.StringVar(value=str(self.state.max_ctx_chars))
        self._add_labeled_entry(gen_box, "temperature", self.temperature_var, 8)
        self._add_labeled_entry(gen_box, "top_p", self.top_p_var, 8)
        self._add_labeled_entry(gen_box, "num_predict", self.num_predict_var, 8)
        self._add_labeled_entry(gen_box, "top_k", self.top_k_var, 5)
        self._add_labeled_entry(gen_box, "mem_pairs", self.mem_pairs_var, 5)
        self._add_labeled_entry(gen_box, "ctx_chars", self.ctx_chars_var, 6)

        advanced_box = ttk.LabelFrame(root, text="Advanced (safety / reasoning / embed)")
        advanced_box.pack(fill=tk.X, pady=(8, 0))
        self.safety_var = tk.BooleanVar(value=self.state.safety_mode)
        self.reasoning_var = tk.BooleanVar(value=self.state.reasoning_nudge)
        self.embed_model_var = tk.StringVar(value=self.state.embed_model)
        ttk.Checkbutton(
            advanced_box, text="Safety mode (refuse cheating/harmful)", variable=self.safety_var
        ).pack(side=tk.LEFT, padx=(6, 10))
        ttk.Checkbutton(
            advanced_box,
            text="Reasoning nudge (check context before answer)",
            variable=self.reasoning_var,
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(advanced_box, text="embed_model").pack(side=tk.LEFT)
        ttk.Entry(advanced_box, textvariable=self.embed_model_var, width=38).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6)
        )

        mid = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        mid.pack(fill=tk.BOTH, expand=True, pady=(10, 6))

        left = ttk.LabelFrame(mid, text="Chat (multi-turn memory)")
        right = ttk.Frame(mid)
        mid.add(left, weight=3)
        mid.add(right, weight=4)

        self.chat = tk.Text(left, wrap=tk.WORD, height=20)
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.configure(state=tk.DISABLED)

        right_split = ttk.PanedWindow(right, orient=tk.VERTICAL)
        right_split.pack(fill=tk.BOTH, expand=True)
        upper = ttk.LabelFrame(right_split, text="Token stats / Retrieved [C#] snippets")
        lower = ttk.LabelFrame(right_split, text="Last prompt sent (debug / report; may be long)")
        right_split.add(upper, weight=3)
        right_split.add(lower, weight=2)
        self.side = tk.Text(upper, wrap=tk.WORD, height=18)
        self.side.pack(fill=tk.BOTH, expand=True)
        self.side.configure(state=tk.DISABLED)
        self.prompt_view = tk.Text(lower, wrap=tk.WORD, height=9)
        self.prompt_view.pack(fill=tk.BOTH, expand=True)
        self.prompt_view.configure(state=tk.DISABLED)

        input_row = ttk.Frame(root)
        input_row.pack(fill=tk.X, pady=(0, 6))
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_row, textvariable=self.input_var)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", lambda _e: self._on_send())
        self.send_btn = ttk.Button(input_row, text="Send", command=self._on_send)
        self.send_btn.pack(side=tk.LEFT, padx=(8, 0))

        bottom = ttk.Frame(root)
        bottom.pack(fill=tk.X)
        ttk.Button(bottom, text="Clear chat + memory", command=self._clear).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Reload docs + reset", command=self._reload).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(bottom, text="Apply params to engine", command=self._apply_params).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.memory_status_var = tk.StringVar(value="")
        ttk.Label(bottom, textvariable=self.memory_status_var).pack(side=tk.LEFT, padx=(12, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.RIGHT)

        self.input_entry.focus_set()

    def _add_labeled_entry(self, parent: ttk.Widget, label: str, var: tk.StringVar, width: int) -> None:
        row = ttk.Frame(parent)
        row.pack(side=tk.LEFT, padx=(8, 0), pady=4)
        ttk.Label(row, text=label).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var, width=width).pack(side=tk.LEFT, padx=(6, 0))

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

    def _set_prompt_debug(self, text: str) -> None:
        self.prompt_view.configure(state=tk.NORMAL)
        self.prompt_view.delete("1.0", tk.END)
        self.prompt_view.insert(tk.END, text)
        self.prompt_view.see(tk.END)
        self.prompt_view.configure(state=tk.DISABLED)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _update_memory_status(self) -> None:
        turns = len(self.companion.memory.turns)
        pairs = turns // 2
        self.memory_status_var.set(
            f"Conversation memory: {pairs} pair(s) / max {self.companion.memory_turns} pair(s)"
        )

    @staticmethod
    def _safe_int(v: str, fallback: int, min_value: int) -> int:
        try:
            parsed = int(v.strip())
        except Exception:
            return fallback
        return parsed if parsed >= min_value else fallback

    @staticmethod
    def _safe_float(v: str, fallback: float, min_value: float, max_value: float) -> float:
        try:
            parsed = float(v.strip())
        except Exception:
            return fallback
        if parsed < min_value or parsed > max_value:
            return fallback
        return parsed

    def _apply_params(self, reset_memory: bool = False) -> None:
        docs_json = self.docs_var.get().strip() or None
        model = self.model_var.get().strip() or self.state.model
        temperature = self._safe_float(self.temperature_var.get(), self.companion.temperature, 0.0, 2.0)
        top_p = self._safe_float(self.top_p_var.get(), self.companion.top_p, 0.0, 1.0)
        num_predict = self._safe_int(self.num_predict_var.get(), self.companion.num_predict, 1)
        top_k = self._safe_int(self.top_k_var.get(), self.companion.top_k, 1)
        memory_turns = self._safe_int(self.mem_pairs_var.get(), self.companion.memory_turns, 1)
        max_ctx_chars = self._safe_int(self.ctx_chars_var.get(), self.companion.max_ctx_chars, 80)
        previous_turns = [] if reset_memory else list(self.companion.memory.turns)

        self.companion = StudyCompanion(
            docs_json=docs_json,
            model=model,
            base_url=self.state.base_url,
            chunk_size=self.state.chunk_size,
            chunk_overlap=self.state.chunk_overlap,
            top_k=top_k,
            memory_turns=memory_turns,
            max_ctx_chars=max_ctx_chars,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
            safety_mode=bool(self.safety_var.get()),
            reasoning_nudge=bool(self.reasoning_var.get()),
        )
        if previous_turns:
            self.companion.memory.turns = previous_turns[-(memory_turns * 2) :]
        self._update_memory_status()
        self._set_status("Params applied")

    def _clear(self) -> None:
        self.companion.memory.turns = []
        self.chat.configure(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.configure(state=tk.DISABLED)
        self._set_side("")
        self._set_prompt_debug("")
        self._update_memory_status()
        self._set_status("Ready")

    def _reload(self) -> None:
        self._apply_params(reset_memory=True)
        self._clear()

    def _on_send(self) -> None:
        if self.is_generating:
            self._set_status("Still generating previous answer, please wait...")
            return
        q = self.input_var.get().strip()
        if not q:
            return
        self.is_generating = True
        self.input_entry.configure(state=tk.DISABLED)
        self.send_btn.configure(state=tk.DISABLED)
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
                    res = self.companion.answer_tfidf(q, top_k=self.companion.top_k)
                else:
                    res = self.companion.answer_embed(
                        q,
                        top_k=self.companion.top_k,
                        embed_model=self.embed_model_var.get().strip() or self.state.embed_model,
                    )
                ans = str(res.get("answer", "")).strip()
                stats = res.get("token_stats")
                retrieved = res.get("retrieved") or []
                prompt_txt = str(res.get("prompt", "")).strip()
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
                self.after(0, lambda: self._set_prompt_debug(prompt_txt))
                self.after(0, self._update_memory_status)
                self.after(0, lambda: self._set_status("Ready"))
                self.after(0, lambda: self._set_generating(False))
            except Exception as e:
                self.after(0, lambda: self._append_chat(f"Assistant: (error) {e}\n"))
                self.after(0, lambda: self._set_status("Ready"))
                self.after(0, lambda: self._set_generating(False))

        threading.Thread(target=run, daemon=True).start()

    def _set_generating(self, generating: bool) -> None:
        self.is_generating = generating
        if generating:
            self.input_entry.configure(state=tk.DISABLED)
            self.send_btn.configure(state=tk.DISABLED)
        else:
            self.input_entry.configure(state=tk.NORMAL)
            self.send_btn.configure(state=tk.NORMAL)
            self.input_entry.focus_set()


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma3:4b")
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument("--docs-json", default=None)
    p.add_argument("--retriever", choices=["baseline", "tfidf", "embed"], default="tfidf")
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

    app = App(
        UiState(
            model=args.model,
            base_url=args.base_url,
            docs_json=args.docs_json,
            retriever=args.retriever,
            memory_turns=args.memory_turns,
            max_ctx_chars=args.max_ctx_chars,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
            safety_mode=args.safety_mode,
            reasoning_nudge=args.reasoning_nudge,
            embed_model=args.embed_model,
        )
    )
    app.mainloop()


if __name__ == "__main__":
    main()
