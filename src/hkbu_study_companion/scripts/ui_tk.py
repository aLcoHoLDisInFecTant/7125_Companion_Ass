from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, ttk

from ..ollama_client import list_local_models
from ..pipeline import StudyCompanion

NO_MODELS_PLACEHOLDER = "No models available"


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
    memory_turns: int
    ctx_chars_each: int
    safety_mode: bool
    reasoning_nudge: bool
    embed_model: str


class App(tk.Tk):
    def __init__(self, state: UiState):
        super().__init__()
        self.state = state
        self.title("HKBU Study Companion")
        self.geometry("1500x860")

        self.available_models: list[str] = []
        self.companion: Optional[StudyCompanion] = None
        self._build_ui()
        self._refresh_models(reset_selection=True)
        self._rebuild_companion(reset_memory=True)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(root)
        top.pack(fill=tk.X)

        self.model_var = tk.StringVar(value=self.state.model)
        self.retriever_var = tk.StringVar(value=self.state.retriever)
        self.docs_var = tk.StringVar(value=self.state.docs_json or "")

        ttk.Label(top, text="Model").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var, state="readonly", width=24)
        self.model_combo.pack(side=tk.LEFT, padx=(6, 6))
        ttk.Button(top, text="Refresh Models", command=self._on_refresh_models).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(top, text="Retriever").pack(side=tk.LEFT)
        ttk.Combobox(
            top,
            textvariable=self.retriever_var,
            values=["baseline", "tfidf", "embed"],
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=(6, 10))
        ttk.Label(top, text="Docs").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.docs_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        ttk.Button(top, text="Browse", command=self._browse_docs).pack(side=tk.LEFT)

        retrieval_cfg = ttk.LabelFrame(root, text="Generation & retrieval (token efficiency / Topic 2)")
        retrieval_cfg.pack(fill=tk.X, pady=(8, 0))

        self.temperature_var = tk.DoubleVar(value=self.state.temperature)
        self.top_p_var = tk.DoubleVar(value=self.state.top_p)
        self.num_predict_var = tk.IntVar(value=self.state.num_predict)
        self.top_k_var = tk.IntVar(value=self.state.top_k)
        self.memory_turns_var = tk.IntVar(value=self.state.memory_turns)
        self.ctx_chars_var = tk.IntVar(value=self.state.ctx_chars_each)

        self._add_spinbox(retrieval_cfg, "temperature", self.temperature_var, 0.0, 2.0, 0.05, 0)
        self._add_spinbox(retrieval_cfg, "top_p", self.top_p_var, 0.0, 1.0, 0.05, 2)
        self._add_spinbox(retrieval_cfg, "num_predict", self.num_predict_var, 16, 2048, 8, 4)
        self._add_spinbox(retrieval_cfg, "top_k", self.top_k_var, 1, 20, 1, 6)
        self._add_spinbox(retrieval_cfg, "mem_pairs", self.memory_turns_var, 1, 20, 1, 8)
        self._add_spinbox(retrieval_cfg, "ctx_chars", self.ctx_chars_var, 80, 2000, 20, 10)

        advanced_cfg = ttk.LabelFrame(root, text="Advanced (safety / reasoning / embed)")
        advanced_cfg.pack(fill=tk.X, pady=(8, 0))

        self.safety_mode_var = tk.BooleanVar(value=self.state.safety_mode)
        self.reasoning_nudge_var = tk.BooleanVar(value=self.state.reasoning_nudge)
        self.embed_model_var = tk.StringVar(value=self.state.embed_model)

        ttk.Checkbutton(
            advanced_cfg,
            text="Safety mode (refuse cheating/harmful)",
            variable=self.safety_mode_var,
        ).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Checkbutton(
            advanced_cfg,
            text="Reasoning nudge (check context before answer)",
            variable=self.reasoning_nudge_var,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(advanced_cfg, text="embed_model").pack(side=tk.LEFT)
        ttk.Entry(advanced_cfg, textvariable=self.embed_model_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6)
        )

        mid = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        mid.pack(fill=tk.BOTH, expand=True, pady=(8, 8))

        left = ttk.Frame(mid)
        right = ttk.Frame(mid)
        mid.add(left, weight=2)
        mid.add(right, weight=2)

        ttk.Label(left, text="Chat (multi-turn memory)").pack(anchor="w")
        self.chat = tk.Text(left, wrap=tk.WORD)
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.configure(state=tk.DISABLED)

        input_row = ttk.Frame(left)
        input_row.pack(fill=tk.X, pady=(6, 0))
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_row, textvariable=self.input_var)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", lambda _e: self._on_send())
        ttk.Button(input_row, text="Send", command=self._on_send).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(right, text="Token stats / Retrieved [C#] snippets").pack(anchor="w")
        self.side = tk.Text(right, wrap=tk.WORD, height=16)
        self.side.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.side.configure(state=tk.DISABLED)

        ttk.Label(right, text="Last prompt sent (debug / report; may be long)").pack(anchor="w")
        self.prompt_debug = tk.Text(right, wrap=tk.WORD, height=10)
        self.prompt_debug.pack(fill=tk.BOTH, expand=True)
        self.prompt_debug.configure(state=tk.DISABLED)

        bottom = ttk.Frame(root)
        bottom.pack(fill=tk.X)

        ttk.Button(bottom, text="Clear chat + memory", command=self._clear).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Reload docs + reset", command=self._reload).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(bottom, text="Apply params to engine", command=self._apply_params).pack(side=tk.LEFT, padx=(8, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.RIGHT)
        self.input_entry.focus_set()

    def _add_spinbox(
        self,
        parent: ttk.LabelFrame,
        label: str,
        variable: tk.Variable,
        from_: float,
        to: float,
        increment: float,
        col: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=0, column=col, padx=(6, 4), pady=6, sticky="w")
        ttk.Spinbox(
            parent,
            textvariable=variable,
            from_=from_,
            to=to,
            increment=increment,
            width=7,
        ).grid(row=0, column=col + 1, padx=(0, 8), pady=6, sticky="w")

    def _browse_docs(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Context File",
            filetypes=[
                ("Supported Context Files", "*.json *.jsonl *.txt *.md *.pdf *.docx"),
                ("JSON/JSONL", "*.json *.jsonl"),
                ("Text/Markdown", "*.txt *.md"),
                ("PDF", "*.pdf"),
                ("Word (.docx)", "*.docx"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.docs_var.set(path)
            # Apply selected knowledge base immediately so users do not need an extra manual reload.
            self._reload()
            self._set_status(f"Loaded docs: {Path(path).name}")

    def _on_refresh_models(self) -> None:
        self._refresh_models(reset_selection=False)
        self._rebuild_companion(reset_memory=False)

    def _refresh_models(self, reset_selection: bool) -> None:
        prev_model = self.model_var.get().strip()
        try:
            models = list_local_models(base_url=self.state.base_url)
        except Exception:
            models = []

        if models:
            self.available_models = models
            self.model_combo.configure(values=models, state="readonly")
            # Keep the current model whenever possible to avoid silent model switches.
            if prev_model and prev_model in models:
                self.model_var.set(prev_model)
            elif self.state.model and self.state.model in models:
                self.model_var.set(self.state.model)
            elif reset_selection or (not prev_model):
                self.model_var.set(models[0])
            elif prev_model not in models:
                self.model_var.set(models[0])
        else:
            self.available_models = []
            self.model_combo.configure(values=[NO_MODELS_PLACEHOLDER], state="disabled")
            self.model_var.set(NO_MODELS_PLACEHOLDER)

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
        self.prompt_debug.configure(state=tk.NORMAL)
        self.prompt_debug.delete("1.0", tk.END)
        self.prompt_debug.insert(tk.END, text)
        self.prompt_debug.see("1.0")
        self.prompt_debug.configure(state=tk.DISABLED)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _memory_pairs(self) -> int:
        if self.companion is None:
            return 0
        return len(self.companion.memory.turns) // 2

    def _status_ready_text(self) -> str:
        if not self._has_usable_model():
            return "No models available. Start Ollama and pull a model, then click Refresh Models."
        return f"Conversation memory: {self._memory_pairs()} pair(s) / max {int(self.memory_turns_var.get())} pair(s)"

    def _has_usable_model(self) -> bool:
        model = self.model_var.get().strip()
        return bool(model) and model != NO_MODELS_PLACEHOLDER

    def _rebuild_companion(self, reset_memory: bool) -> None:
        if not self._has_usable_model():
            self.companion = None
            self._set_status(self._status_ready_text())
            return

        old_turns = []
        if self.companion is not None and not reset_memory:
            old_turns = list(self.companion.memory.turns)

        self.companion = StudyCompanion(
            docs_json=self.docs_var.get().strip() or None,
            model=self.model_var.get().strip() or self.state.model,
            base_url=self.state.base_url,
            chunk_size=int(self.state.chunk_size),
            chunk_overlap=int(self.state.chunk_overlap),
            top_k=int(self.top_k_var.get()),
            memory_turns=int(self.memory_turns_var.get()),
            temperature=float(self.temperature_var.get()),
            top_p=float(self.top_p_var.get()),
            num_predict=int(self.num_predict_var.get()),
            ctx_chars_each=int(self.ctx_chars_var.get()),
            safety_mode=bool(self.safety_mode_var.get()),
            reasoning_nudge=bool(self.reasoning_nudge_var.get()),
        )
        if old_turns:
            max_turns = int(self.memory_turns_var.get())
            self.companion.memory.turns = old_turns[-2 * max_turns :]
        self._set_status(self._status_ready_text())

    def _clear(self) -> None:
        if self.companion is not None:
            self.companion.memory.turns = []
        self.chat.configure(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.configure(state=tk.DISABLED)
        self._set_side("")
        self._set_prompt_debug("")
        self._set_status(self._status_ready_text())

    def _reload(self) -> None:
        self._rebuild_companion(reset_memory=True)
        self._clear()

    def _apply_params(self) -> None:
        self._rebuild_companion(reset_memory=False)
        self._set_status(self._status_ready_text())

    def _on_send(self) -> None:
        q = self.input_var.get().strip()
        if not q:
            return
        if self.companion is None:
            self._append_chat("Assistant: (error) No local model is available. Please start Ollama, pull a model, and click Refresh Models.\n")
            self._set_status(self._status_ready_text())
            return
        self.input_var.set("")
        self._append_chat(f"User: {q}")
        self._set_status("Generating...")
        self._set_side("")

        def run() -> None:
            try:
                mode = self.retriever_var.get().strip()
                top_k = int(self.top_k_var.get())
                embed_model = self.embed_model_var.get().strip() or self.state.embed_model

                if mode == "baseline":
                    res = self.companion.answer_baseline(q)
                elif mode == "tfidf":
                    res = self.companion.answer_tfidf(q, top_k=top_k)
                else:
                    res = self.companion.answer_embed(q, top_k=top_k, embed_model=embed_model)

                ans = str(res.get("answer", "")).strip()
                stats = res.get("token_stats")
                retrieved = res.get("retrieved") or []
                prompt_text = str(res.get("prompt", ""))

                side_lines = [f"token_stats: {stats}", "", "retrieved:"]
                if not retrieved:
                    side_lines.append("(None)")
                else:
                    for i, r in enumerate(retrieved, start=1):
                        t = str(r.get("text", "")).strip().replace("\n", " ")
                        if len(t) > 380:
                            t = t[:380].rstrip() + "..."
                        score = r.get("score")
                        score_str = f"{float(score):.6f}" if score is not None else "NA"
                        side_lines.append(
                            f"[C{i}] score={score_str} | {r.get('doc_id')} | {r.get('chunk_id')} | {r.get('title')}"
                        )
                        side_lines.append(t)
                        side_lines.append("")

                self.after(0, lambda: self._append_chat(f"Assistant: {ans}\n"))
                self.after(0, lambda: self._set_side("\n".join(side_lines)))
                self.after(0, lambda: self._set_prompt_debug(prompt_text))
                self.after(0, lambda: self._set_status(self._status_ready_text()))
            except Exception as e:
                self.after(0, lambda: self._append_chat(f"Assistant: (error) {e}\n"))
                self.after(0, lambda: self._set_status(self._status_ready_text()))

        threading.Thread(target=run, daemon=True).start()


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma3:4b")
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument(
        "--docs-json",
        default=None,
        help="Context file path (.json/.jsonl/.txt/.md/.pdf/.docx)",
    )
    p.add_argument("--retriever", choices=["baseline", "tfidf", "embed"], default="tfidf")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=220)
    p.add_argument("--chunk-overlap", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--num-predict", type=int, default=220)
    p.add_argument("--memory-turns", type=int, default=4)
    p.add_argument("--ctx-chars", type=int, default=420)
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
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
            memory_turns=args.memory_turns,
            ctx_chars_each=args.ctx_chars,
            safety_mode=args.safety_mode,
            reasoning_nudge=args.reasoning_nudge,
            embed_model=args.embed_model,
        )
    )
    app.mainloop()


if __name__ == "__main__":
    main()
