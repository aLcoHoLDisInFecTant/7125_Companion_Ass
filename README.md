# HKBU Study Companion (Local, Ollama + Hand-Rolled RAG)

## 使用教程（Windows / 实验室环境友好）

本项目是“纯 Python + 手写 RAG + Ollama generate”的最小可复现实验框架：
- 不使用 `ollama.chat`
- 不使用 LangChain
- 评估脚本会打印 Ollama 返回的 token 统计（`prompt_eval_count` / `eval_count`）

### 0) 前置条件

1) 本机已安装并启动 Ollama（默认监听 `http://localhost:11434`）。
2) 本机已拉取至少一个可用模型。你当前环境里已存在的示例模型通常是：

```bash
ollama list
ollama pull gemma3:4b
```

如果你本地也有 `qwen3:9b`，也可以用它；命令行参数里的 `--model` 只要填写 `ollama list` 里出现的名字即可。

### 1) 安装依赖（两种方式任选其一）

#### 方式 A（推荐：不安装包，直接运行源码）

```powershell
cd d:\neuro-like\7125Core\hkbu-study-companion
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
```

#### 方式 B（可选：可编辑安装）

```powershell
cd d:\neuro-like\7125Core\hkbu-study-companion
pip install -r requirements.txt
pip install -e .
```

### 2) 放置你的知识库 JSON/JSONL

推荐放在项目的 `data/` 目录下，例如：
- `data/docs.example.json`（JSON 数组示例，项目自带）
- `data/rag_knowledge_base.jsonl`（你自己的 JSONL）

支持的格式：

- JSON 数组（推荐）：
  ```json
  [
    {"doc_id":"DOC1","title":"...","text":"..."}
  ]
  ```
- JSONL（每行一个 JSON 对象）：
  ```json
  {"doc_id":"DOC1","source_file":"x.pdf","category":"Preamble","clean_markdown_content":"...","summary":"..."}
  ```

字段映射规则（你不用改原始字段也能跑）：
- `text` 优先；如果没有，会依次尝试：`clean_markdown_content` → `content` → `summary`
- `title` 若缺省，会用 `source_file | category` 自动拼一个可读标题

### 3) 运行评估（Baseline vs Lexical RAG vs Neural RAG）

#### 3.1 用你自己的 JSONL（示例：rag_knowledge_base.jsonl）

```powershell
cd d:\neuro-like\7125Core\hkbu-study-companion
$env:PYTHONPATH = "$PWD\src"
python -m hkbu_study_companion.scripts.run_eval `
  --docs-json "d:\neuro-like\7125Core\hkbu-study-companion\data\rag_knowledge_base.jsonl" `
  --query "上课教室在哪里" `
  --mode all `
  --model gemma3:4b
```

如果你只想跑其中一种模式：

```powershell
python -m hkbu_study_companion.scripts.run_eval --mode tfidf  --docs-json "data\rag_knowledge_base.jsonl" --query "..." --model gemma3:4b
python -m hkbu_study_companion.scripts.run_eval --mode baseline --docs-json "data\rag_knowledge_base.jsonl" --query "..." --model gemma3:4b
python -m hkbu_study_companion.scripts.run_eval --mode embed   --docs-json "data\rag_knowledge_base.jsonl" --query "..." --model gemma3:4b
```

### 4) 启动对话 CLI（可交互提问）

```powershell
cd d:\neuro-like\7125Core\hkbu-study-companion
$env:PYTHONPATH = "$PWD\src"
python -m hkbu_study_companion.scripts.chat_cli `
  --model gemma3:4b `
  --docs-json "d:\neuro-like\7125Core\hkbu-study-companion\data\rag_knowledge_base.jsonl" `
  --retriever tfidf
```

进入后会出现 `>`，直接输入问题即可。

CLI 内置命令：
- `/mode baseline|tfidf|embed` 切换检索方式
- `/clear` 清空对话记忆
- `/exit` 退出

### 5) 启动简单 UI（Tkinter，本地无额外依赖）

如果你更希望用窗口界面输入问题、查看 token 统计和检索片段，可以启动 Tkinter UI：

```powershell
cd d:\neuro-like\7125Core\hkbu-study-companion
$env:PYTHONPATH = "$PWD\src"
python -m hkbu_study_companion.scripts.ui_tk `
  --model gemma3:4b `
  --docs-json "d:\neuro-like\7125Core\hkbu-study-companion\data\rag_knowledge_base.jsonl" `
  --retriever tfidf
```

UI 功能：
- 左侧：对话窗口（User/Assistant）
- 右侧：本次回答的 token 统计与检索到的上下文片段
- 顶部：可切换 retriever（baseline/tfidf/embed）与选择 docs 文件

## Notes

- This project uses only Ollama generate (HTTP `/api/generate` or `ollama.generate` if installed).
- No `ollama.chat`, no LangChain.
- Embedding retrieval uses `sentence-transformers` and will use CUDA if available (`torch.cuda.is_available()`).

### Windows 下 Embedding 常见问题（symlink）

如果你在 `--mode embed` 或 `/mode embed` 时看到 symlink/缓存相关错误（例如 WinError 14007），常见原因是 Windows 对符号链接限制导致 HuggingFace 缓存失败。

解决方向：
- 开启 Windows Developer Mode（允许创建 symlink）
- 或使用具备 symlink 权限的运行方式
- 或提前把 embedding 模型下载到本地目录，然后把 `--embed-model` 指向本地路径

在无法解决前，建议先用 `tfidf` 跑通完整流程（最稳定）。
