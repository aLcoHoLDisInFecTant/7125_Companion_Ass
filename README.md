# HKBU Study Companion (Local, Ollama + Hand-Rolled RAG)

This project is a minimal and reproducible local assistant for HKBU course Q&A and study planning.
It uses raw generation with Ollama and a hand-rolled RAG pipeline.

- Uses `ollama.generate(...)` or HTTP `POST /api/generate`
- Does **not** use `ollama.chat`
- Does **not** use LangChain
- Supports baseline, lexical RAG (TF-IDF), and embedding RAG
- Exposes token-usage stats (`prompt_eval_count`, `eval_count`)

## 1) Prerequisites

1. Install and start Ollama locally (default endpoint: `http://localhost:11434`)
2. Pull at least one local model:

```bash
ollama list
ollama pull gemma3:4b
```

You can also use `qwen3:9b` or any compatible local model.

## 2) Install Dependencies

### Option A (recommended): run from source

```powershell
cd <project-root>
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
```

### Option B: editable install

```powershell
cd <project-root>
pip install -r requirements.txt
pip install -e .
```

Available commands after installation:

```powershell
hkbu-eval --help
hkbu-chat --help
hkbu-ui --help
```

## 3) Data Files (JSON / JSONL)

Place your files under `data/`, for example:

- `data/docs.example.json`
- `data/rag_knowledge_base.jsonl`

Supported fields:

- Preferred text field: `text`
- Fallback fields: `clean_markdown_content` -> `content` -> `summary`
- If `title` is missing, it is composed from `source_file | category`

## 4) Run Evaluation

```powershell
cd <project-root>
$env:PYTHONPATH = "$PWD\src"
python -m hkbu_study_companion.scripts.run_eval `
  --docs-json "$PWD\data\rag_knowledge_base.jsonl" `
  --query "What is the late submission policy for COMP4146?" `
  --mode all `
  --model gemma3:4b
```

Single mode examples:

```powershell
python -m hkbu_study_companion.scripts.run_eval --mode baseline --docs-json "data\rag_knowledge_base.jsonl" --query "..." --model gemma3:4b
python -m hkbu_study_companion.scripts.run_eval --mode tfidf --docs-json "data\rag_knowledge_base.jsonl" --query "..." --model gemma3:4b
python -m hkbu_study_companion.scripts.run_eval --mode embed --docs-json "data\rag_knowledge_base.jsonl" --query "..." --model gemma3:4b
```

## 5) Run Interactive CLI

```powershell
cd <project-root>
$env:PYTHONPATH = "$PWD\src"
python -m hkbu_study_companion.scripts.chat_cli `
  --model gemma3:4b `
  --docs-json "$PWD\data\rag_knowledge_base.jsonl" `
  --retriever tfidf
```

CLI commands:

- `/mode baseline|tfidf|embed`
- `/clear`
- `/exit`

## 6) Run Tkinter UI

```powershell
cd <project-root>
$env:PYTHONPATH = "$PWD\src"
python -m hkbu_study_companion.scripts.ui_tk `
  --model gemma3:4b `
  --docs-json "$PWD\data\rag_knowledge_base.jsonl" `
  --retriever tfidf `
  --memory-turns 4 `
  --max-ctx-chars 420 `
  --safety-mode `
  --reasoning-nudge
```

UI highlights:

- Chat panel with multi-turn memory
- Retrieved snippets and token stats
- Last prompt viewer (for debugging and reporting)
- Runtime controls:
  - `temperature`, `top_p`, `num_predict`, `top_k`, `mem_pairs`, `ctx_chars`
  - `Safety mode`, `Reasoning nudge`, `embed_model`
  - `Apply params to engine`, `Reload docs + reset`

## 7) Notes

- Embedding retrieval uses `sentence-transformers` and can use CUDA automatically
- On Windows, embedding model downloads may require Developer Mode / symlink support
- If embedding setup fails, use `tfidf` mode first to keep the pipeline running
