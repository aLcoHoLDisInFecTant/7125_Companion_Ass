# HKBU Study Companion (Local, Ollama + Hand-Rolled RAG)

This project is a minimal, reproducible framework built with pure Python, hand-rolled RAG, and Ollama generation.

- Uses `ollama.generate` or HTTP `POST /api/generate`
- Does not use `ollama.chat`
- Does not use LangChain
- Evaluation scripts print token stats (`prompt_eval_count` / `eval_count`)

## 0) Prerequisites

1) Install and start Ollama locally (default: `http://localhost:11434`).
2) Pull at least one usable model:

```bash
ollama list
ollama pull gemma3:4b
```

You can also use other local models such as `qwen3:9b`.

## 1) Install Dependencies

### Option A (recommended): run source directly

```powershell
cd e:\hkbu\comp7125-prompt engineering\groupproject\7125_Companion_Ass
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
```

### Option B: editable package install

```powershell
cd e:\hkbu\comp7125-prompt engineering\groupproject\7125_Companion_Ass
pip install -r requirements.txt
pip install -e .
```

After installation, these commands are available:

```powershell
hkbu-eval --help
hkbu-chat --help
hkbu-ui --help
```

## 2) Prepare Knowledge Base Files (JSON/JSONL)

Put your docs under `data/`, for example:

- `data/docs.example.json` (sample JSON array)
- `data/rag_knowledge_base.jsonl` (your JSONL knowledge base)

Supported formats:

- JSON array:
  ```json
  [
    {"doc_id":"DOC1","title":"...","text":"..."}
  ]
  ```
- JSONL (one JSON object per line):
  ```json
  {"doc_id":"DOC1","source_file":"x.pdf","category":"Preamble","clean_markdown_content":"...","summary":"..."}
  ```

Field mapping:

- `text` is preferred
- fallback order: `clean_markdown_content` -> `content` -> `summary`
- if `title` is missing, it is auto-composed from `source_file | category`

## 3) Run Evaluation (Baseline vs Lexical RAG vs Neural RAG)

```powershell
cd e:\hkbu\comp7125-prompt engineering\groupproject\7125_Companion_Ass
$env:PYTHONPATH = "$PWD\src"
python -m hkbu_study_companion.scripts.run_eval `
  --docs-json "data\rag_knowledge_base.jsonl" `
  --query "What is the late submission policy?" `
  --mode all `
  --model gemma3:4b
```

Run a single mode:

```powershell
python -m hkbu_study_companion.scripts.run_eval --mode tfidf --docs-json "data\rag_knowledge_base.jsonl" --query "..." --model gemma3:4b
python -m hkbu_study_companion.scripts.run_eval --mode baseline --docs-json "data\rag_knowledge_base.jsonl" --query "..." --model gemma3:4b
python -m hkbu_study_companion.scripts.run_eval --mode embed --docs-json "data\rag_knowledge_base.jsonl" --query "..." --model gemma3:4b
```

If installed with Option B:

```powershell
hkbu-eval --docs-json "data\rag_knowledge_base.jsonl" --query "..." --mode all --model gemma3:4b
```

## 4) Start Interactive CLI Chat

```powershell
cd e:\hkbu\comp7125-prompt engineering\groupproject\7125_Companion_Ass
$env:PYTHONPATH = "$PWD\src"
python -m hkbu_study_companion.scripts.chat_cli `
  --model gemma3:4b `
  --docs-json "data\rag_knowledge_base.jsonl" `
  --retriever tfidf
```

CLI commands:

- `/mode baseline|tfidf|embed`
- `/clear`
- `/exit`

If installed with Option B:

```powershell
hkbu-chat --model gemma3:4b --docs-json "data\rag_knowledge_base.jsonl" --retriever tfidf
```

## 5) Start Local GUI (Tkinter)

```powershell
cd e:\hkbu\comp7125-prompt engineering\groupproject\7125_Companion_Ass
$env:PYTHONPATH = "$PWD\src"
python -m hkbu_study_companion.scripts.ui_tk `
  --model gemma3:4b `
  --docs-json "data\rag_knowledge_base.jsonl" `
  --retriever tfidf
```

GUI features:

- Auto-discovered model dropdown (with `Refresh Models` button)
- Placeholder and disabled model control when no model is available
- Left panel: multi-turn conversation
- Right panel: token stats, retrieved snippets, and last prompt (debug view)
- Top controls: retriever, docs selector, generation/retrieval parameters
- Advanced controls: `safety_mode`, `reasoning_nudge`, and `embed_model`
- Utility buttons: `Clear chat + memory`, `Reload docs + reset`, `Apply params to engine`

If installed with Option B:

```powershell
hkbu-ui --model gemma3:4b --docs-json "data\rag_knowledge_base.jsonl" --retriever tfidf
```

## Notes

- Uses only Ollama generate API (`/api/generate`) or `ollama.generate`
- Embedding retrieval uses `sentence-transformers` and can use CUDA when available (`torch.cuda.is_available()`)
- Conversation memory keeps pending turns correctly and trims by `mem_pairs`
- Empty model responses are guarded with a non-empty fallback message

## Troubleshooting

- If the answer says context is insufficient, check whether your query matches the loaded docs topic.
- `top_k` returns candidates, but candidates may still be semantically irrelevant.
- If no model is listed, start Ollama and pull a model, then click `Refresh Models`.
- If model output is unstable, retry once, switch model, or reduce `top_k`/`ctx_chars`.

### Windows embedding note (symlink)

If embedding mode fails with symlink/cache errors (for example WinError 14007), common fixes:

- Enable Windows Developer Mode (allows symlink creation)
- Run with permissions that allow symlink creation
- Pre-download embedding model and pass a local path via `--embed-model`

If unresolved, use `tfidf` first to complete the workflow.
