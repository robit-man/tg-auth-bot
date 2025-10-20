# Cognition Framework — API Documentation

A modular Python framework for **agentic LLM workflows** with:

* A clean **Ollama** chat wrapper (`cognition.py`)
* **Hybrid memory** (vector + relational) with provenance (`memory.py`)
* **Tool calling** with a standardized schema and a production-grade **web search + scraping** tool (`tools.py`)
* Clause-aware, hybrid **RAG** with neighbor “bleed”, optional **PDF page vision extraction**, and automatic ingestion pipeline (`context.py`)
* **Online learning** for routing and parameter tuning (LinUCB policy + param bandits) with HITL feedback (`policy.py`)
* A full **interactive demo app** showcasing routing, HITL correction, A/B prompt tuning, Tools, RAG, and Memory (`main.py`)

> All modules are **backward compatible** with existing databases; schema migrations are automatic.

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [Configuration](#configuration)
3. [Module: `cognition.py`](#module-cognitionpy)
4. [Module: `memory.py`](#module-memorypy)
5. [Module: `tools.py`](#module-toolspy)
6. [Module: `context.py` (Knowledge/RAG)](#module-contextpy-knowledgerag)
7. [Module: `policy.py`](#module-policypy)
8. [App: `main.py` (Interactive Demos + HITL)](#app-mainpy-interactive-demos--hitl)
9. [Extending the System](#extending-the-system)
10. [Troubleshooting & Tips](#troubleshooting--tips)

---

## Quickstart

```bash
# 1) Ensure Ollama is installed & running, and you have a chat model pulled (e.g., gemma3:4b) and an embed model
#    Pull examples:
#    ollama pull gemma3:4b
#    ollama pull nomic-embed-text
#    ollama pull llava   # optional, for vision on PDF pages

# 2) Run the app — it will create a venv, install deps, and re-exec itself
python main.py
```

* Drop documents into `./data` (PDF/MD/TXT/HTML/DOCX).
* Choose “Knowledge: Ask the library (RAG)” to query them.
* Choose “Tool: Search Internet” to run a headful browser scraping/search flow.
* Route selection learns via your ratings and corrections.

---

## Configuration

### `config.json`

Auto-created/updated. Keys:

```json
{
  "model": "gemma3:4b",
  "temperature": 0.2,
  "stream": false,
  "global_system_prelude": "",
  "embed_model": "nomic-embed-text",
  "vision_model": "llava"   // optional; used by RAG page-vision
}
```

* Env overrides:

  * `OLLAMA_MODEL` → default chat model
  * `OLLAMA_EMBED_MODEL` → embedding model
  * `OLLAMA_VISION_MODEL` → vision model for PDF pages

### System prompts & schemas

* `system.json`: base prompts & JSON Schemas
* `system_modified.json`: user-modified prompts/schemas (HITL/A/B tuned)

Both are auto-managed by `main.py` (you can also edit via the built-in prompt editor).

---

## Module: `cognition.py`

### Overview

`CognitionChat` wraps **Ollama** chat/streaming and adds higher-level “modalities”:

* **Raw chat**
* **Decide from options** (classification / router)
* **Structured JSON** (schema-validated outputs)
* **Produce system message** (prompt authoring)

`CognitionConfig` holds model/runtime defaults and persists in `config.json`.

### Key Classes

#### `CognitionConfig`

```python
from cognition import CognitionConfig

cfg = CognitionConfig.load()                 # load from config.json (or defaults)
cfg.model = "gemma3:4b"
cfg.embed_model = "nomic-embed-text"
cfg.temperature = 0.2
cfg.stream = False
cfg.global_system_prelude = ""
cfg.save()
```

Fields:

* `model: str`
* `temperature: float`
* `stream: bool`
* `global_system_prelude: str`
* `embed_model: str`
* `vision_model: Optional[str]` (optional; used by `context.py`)

#### `CognitionChat`

```python
from cognition import CognitionChat, CognitionConfig

cfg = CognitionConfig.load()
cog = CognitionChat(cfg)

# Set a system message
cog.set_system("You are brief and helpful. Cite sources if provided in context.")

# Add context (strings or dicts). Dicts can include bookkeeping info; strings go verbatim.
cog.add_context([
    "Project codename: Aurora. Deadline next Friday.",
    {"meta": {"ticket": "AUR-123"}},
])

# 1) Raw chat
resp = cog.raw_chat("What's our project and deadline?", stream=False)
print(resp)

# 2) Decide from options
choice = cog.decide_from_options(
    question="I want to search the web for nearby coffee roasters.",
    options=["Raw chat", "Tool: Search Internet", "Knowledge: Ask the library (RAG)"],
    return_index=False,
    stream=False,
)
print("Router choice:", choice)

# 3) Structured JSON (validated against a schema)
todo_schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "TodoItem",
    "type": "object",
    "required": ["title", "priority", "tags"],
    "properties": {
        "title": {"type": "string"},
        "priority": {"type": "integer", "minimum": 1, "maximum": 5},
        "tags": {"type": "array", "items": {"type": "string"}}
    },
    "additionalProperties": False
}
todo = cog.structured_json("Plan a kickoff meeting for Aurora.", json_schema=todo_schema, stream=False)
print(todo)

# 4) Produce system message (prompt authoring)
sysmsg = cog.produce_system_message("Generate a system prompt for answering concisely with bullet points.")
print(sysmsg)
```

**Debug-fallback:** If a modality requires inputs you didn’t provide (e.g., missing schema), the wrapper injects a debug system message and echoes arguments so the model can suggest corrections (useful in loops).

---

## Module: `memory.py`

### Overview

`MemoryStore` is a hybrid **SQLite** memory with **vector embeddings** and **provenance** links:

* Store **user/assistant** turns and tool outputs
* Recall via **similarity**, **timestamp**, **tags**
* Link **invocations** (a call/run) to memories (`produced`) and to **recalled** entries
* WAL mode, busy timeouts, thread-safe writes

### Typical Turn Flow

```python
from memory import MemoryStore

mem = MemoryStore(db_path="memory.db", embed_model="nomic-embed-text")

session_id = "demo-session-001"

# 1) User says something
user_text = "We committed to support Aurora by Friday. Remind me of risks."
user_mem_id = mem.add_memory(
    session_id=session_id,
    role="user",
    content=user_text,
    source="user_input",
    tags=["project:aurora", "milestone:deadline"]
)

# 2) Recall recent context to prime the model
recalled = mem.recall_for_context(session_id, query_text=user_text, k=8)

# 3) Model generates a response (not shown)
assistant_text = "Risks: unclear scope, resource contention, vendor delays."

# 4) Track an invocation (this step/turn)
inv_id = mem.add_invocation(session_id, mode="raw_chat", user_input=user_text, meta={"notes": "first pass"})

# Link produced + recalled
asst_mem_id = mem.add_memory(session_id, "assistant", assistant_text, "cog_output", tags=["mode:raw_chat"])
mem.link_invocation_memory(inv_id, user_mem_id, relation="produced")
mem.link_invocation_memory(inv_id, asst_mem_id, relation="produced")
for r in recalled:
    mem.link_invocation_memory(inv_id, r.id, relation="recalled")
```

**API Highlights**

* `add_memory(session_id, role, content, source, tags=[]) -> str`
* `recall_for_context(session_id, query_text, k=8) -> List[Memory]`
* `add_invocation(session_id, mode, user_input, meta={}) -> str`
* `link_invocation_memory(invocation_id, memory_id, relation="recalled"|"produced") -> None`

---

## Module: `tools.py`

### Overview

A standardized **tool calling** layer with:

* `Tools.search_internet(...)`: **DuckDuckGo → Selenium headful browse → deep scrape → optional LLM summary**
* Production-ready browser management: locate or auto-install **ChromeDriver**, headful browsing, waits, input helpers
* **Obfuscated HTTP scraping** path with realistic headers/cookies to minimize 403s
* Optional **auxiliary inference** step to summarize scraped pages via your chat model

> We recommend **headless=False** for best compatibility with modern sites.

### Quick Use

```python
from tools import Tools

# One-shot search + scrape
results = Tools.search_internet(
    topic="best static site generators 2025",
    num_results=5,
    wait_sec=2,
    deep_scrape=True,
    summarize=True,     # Uses auxiliary_inference(...) under the hood
    bs4_verbose=False,  # True = store raw HTML; False = visible text only
    headless=False
)

for i, r in enumerate(results, 1):
    print(f"[{i}] {r['title']}\n{r['url']}\nSnippet: {r['snippet']}\nSummary: {r.get('aux_summary','')[:300]}\n")
```

### Browser Helpers

```python
Tools.open_browser(headless=False, force_new=True)
Tools.navigate("https://example.com")
Tools.click("button#start")
Tools.input("input#search", "llm agents")
html = Tools.get_html()
Tools.screenshot("page.png")
Tools.close_browser()
```

Internals worth noting:

* `_find_system_chromedriver()` heuristics
* `_wait_for_ready(...)`, `_first_present(...)`, `_visible_and_enabled(...)`
* `bs4_scrape(url, verbosity=False)` with realistic headers, cookie jar, minor randomization
* `auxiliary_inference(prompt=...)` uses your chat model to summarize scraped content

---

## Module: `context.py` (Knowledge/RAG)

### Overview

`KnowledgeStore` is a robust, clause-aware, hybrid **RAG** engine:

* **Automatic ingestion** from `./data` (PDF/MD/TXT/HTML/DOCX)
* **Clause detection** (e.g., “7.5.3 Control of documented information”) and clause-aware chunking
* **Neighbor “bleed”**: include ±N adjacent chunks for richer local context
* Optional per-page **vision extraction** for PDFs:

  * render page → PNG
  * run **Ollama multimodal** model (e.g., `llava`) to read tables/figures/low-OCR text
* Hybrid retrieval:

  * Vector similarity (embeddings)
  * Optional **FTS5** (if available in your SQLite build)
  * **Reciprocal Rank Fusion** of vector + keyword
* **Backward-compatible DB migrations** (safe on older `knowledge.db`)

### Schema (simplified)

* `documents(id, path, checksum, title, type, size, mtime, tags_json, added_ts, updated_ts, status, meta_json)`
* `doc_tags(doc_id, tag)`
* `clauses(id, doc_id, clause_code, title, text, page_start, page_end)`
* `chunks(id, doc_id, chunk_index, text, page_no, ts, clause_code, heading, chunk_idx, prev_chunk_id, next_chunk_id, page_start, page_end, modality, resource_path)`
* `chunk_vectors(chunk_id, embedding, dim, norm)`
* `chunks_fts(chunk_id, content, clause_code)` *(optional, requires FTS5)*
* `invocations(id, session_id, query, ts, meta_json)`
* `invocation_chunks(invocation_id, chunk_id, relation)`

### Ingestion

```python
from context import KnowledgeStore
from pathlib import Path

store = KnowledgeStore(db_path="knowledge.db", data_dir="data", embed_model="nomic-embed-text")

# One-shot ingest (concurrent), with optional force & vision:
stats = store.ingest_all_in_data(
    concurrent=True,
    max_workers=4,
    force=False,                # True to rebuild even unchanged files
    vision=True,                # Render each PDF page → PNG and run vision model
    vision_model="llava"        # or via OLLAMA_VISION_MODEL env var
)
print(stats)  # {'total': N, 'ingested': x, 'skipped': y, 'failed': z}

# Background watcher (periodic rescans):
store.start_auto_ingest_watcher(interval_sec=30, concurrent=True, max_workers=4, force=False, vision=False)
```

Supported types: `.pdf`, `.md`, `.markdown`, `.txt`, `.html`, `.htm`, `.docx`.

### Search & Build Context

```python
# Basic vector search
top = store.search(query_text="AS9100 clause 7.5.3 documented information", k=8, tags=None)

# Hybrid search with neighbor bleed and RRF fusion (recommended)
top = store.search_with_bleed(
    query_text="nonconformance control per AS9131 sampling requirements",
    k=8,
    bleed=1,        # include ±1 neighbor chunk for context
    tags=None
)

# Turn into LLM-ready context snippets (with proper citations)
snips, chunk_ids = store.build_context_snippets(top)
for s in snips:
    print(s, "\n")

# Track invocation + provenance
inv_id = store.add_invocation(session_id="sess-001", query="audit evidence for 7.5.3", meta={"k": 8, "bleed": 1})
for cid in chunk_ids:
    store.link_invocation_chunk(invocation_id=inv_id, chunk_id=cid)
```

Snippets look like:

```
[source: ISO 9001:2015 | clause 7.5.3 | p.10 | #a1b2c3d4 | score=0.812]
…chunk text…
```

If vision was used, modality is included and the snippet can include an image hint.

---

## Module: `policy.py`

### Overview

Online learning to improve:

* **Routing** (which demo/path to take) via a **LinUCB** contextual bandit
* **Parameters** (e.g., RAG `top_k`, `bleed`, or choosing **base** vs **modified** system prompts) via multi-armed **ParamBandit**
* Lightweight **FeatureExtractor** for user intents
* **FeedbackLog** and **heavy failure** detection → triggers A/B prompt tuning

### Typical Usage

```python
from policy import PolicyManager

policy = PolicyManager(db_path="policy.db")

# 1) Routing: choose among actions given features
feats = policy.features.from_intent("I want to ask questions about my PDFs")
action, scores = policy.policy.select(
    actions=["raw_chat", "structured_json", "knowledge_rag", "tool_search_internet"],
    features=feats
)

# 2) Update with reward in [0,1]
policy.policy.update(action, feats, reward=0.8)

# 3) Param bandits (e.g., RAG top_k)
arm = policy.params.select("rag.top_k", arms=["5", "8", "12"])
# ... run with chosen arm ...
policy.params.update("rag.top_k", arm, reward=0.9)

# 4) Heavy failure monitor
if policy.feedback.heavy_failure(action, n=5, threshold=0.25, min_count=3):
    print("Consider A/B testing the system prompt for this action.")
```

---

## App: `main.py` (Interactive Demos + HITL)

### What it Provides

* **Venv bootstrap** + deps install
* **Menu router** with LLM & policy suggestions
* **HITL correction** using natural language (LLM normalizes free text → exact route)
* **Prompt A/B** and **editor** (stores modified prompts in `system_modified.json`)
* Three core demos:

  1. **Raw chat** (with memory recall)
  2. **Structured JSON** (TodoItem sample)
  3. **Produce system message**
  4. **Tool: Search Internet** (plans JSON args, then executes `Tools.search_internet`)
  5. **Knowledge: Ask the library (RAG)** (hybrid search + bleed; optional force re-ingest and vision each run)

### Running

```bash
python main.py
```

* At the **decision stage**, describe your goal.
* The router proposes a path. You can:

  * `Y` to proceed
  * `no` to correct (free text allowed; LLM normalizes to a valid route)
  * Type another route name directly (free text allowed)
  * `edit` to open the prompt/schema editor

**RAG run** prompts for:

* Force re-ingest unchanged files? (`y/N`)
* Enable page vision extraction? (`y/N`) → uses `OLLAMA_VISION_MODEL` or `"llava"`

**Reward collection**: after each run, provide a 1–5 rating; the system updates the router and param bandits.

---

## Extending the System

### Add a New Tool

1. Implement your function in `tools.py`:

```python
class Tools:
    @staticmethod
    def my_tool(arg1: str, n: int = 3) -> dict:
        """Describe purpose and return shape in the docstring."""
        # do work...
        return {"ok": True, "items": [...]} 
```

2. Add a **JSON Schema** for its args in `system.json → schemas`.
3. In `main.py`, add a new **menu option**, plan via `CognitionChat.structured_json` against your schema, then call the tool with the returned args.

### Add a New Route

* Append to `DEMO_OPTIONS` and `ACTION_MAP`.
* Implement a `run_*` function following existing patterns:

  * use `choose_sysmsg_variant(...)` for base/modified prompt
  * construct context from memory or RAG
  * log invocation + link recalled/produced resources
  * gather a reward and update policy

---

## Troubleshooting & Tips

* **SQLite FTS5**: If your SQLite build lacks FTS5, `context.py` disables FTS automatically. Vector search still works.
* **Schema migrations**: Existing `knowledge.db` is migrated in-place (adds new columns before new indexes).
* **403s on scraping**: Use **headful** browsing (`headless=False`), rely on `Tools.search_internet` (Selenium + realistic HTTP headers in `bs4_scrape`). Some sites still block automated agents.
* **ChromeDriver**: `tools.py` tries Selenium-Manager, snap chromedriver, PATH, webdriver-manager, and last-resort snap install.
* **Models**:

  * Chat: set `OLLAMA_MODEL` (e.g., `gemma3:4b`)
  * Embedding: set `OLLAMA_EMBED_MODEL` (e.g., `nomic-embed-text`)
  * Vision: set `OLLAMA_VISION_MODEL` (e.g., `llava`)
* **Memory hygiene**: Only store **final stage outputs** + **user input** (the top-level decision stage is not stored).
* **HITL**: Natural-language corrections (even rough) are normalized by the LLM to one of the exact menu options.

---

## End-to-End Example (Programmatic)

```python
from cognition import CognitionConfig, CognitionChat
from memory import MemoryStore
from context import KnowledgeStore

# Load config and set models
cfg = CognitionConfig.load()
cfg.model = "gemma3:4b"
cfg.embed_model = "nomic-embed-text"
cfg.save()

# Memory store for a session
session = "example-session-001"
mem = MemoryStore("memory.db", embed_model=cfg.embed_model)

# Knowledge store + ingest docs in ./data (with vision)
store = KnowledgeStore("knowledge.db", "data", embed_model=cfg.embed_model)
store.ingest_all_in_data(concurrent=True, vision=True, vision_model="llava")

# Ask a RAG question
q = "How does AS9100 handle control of documented information?"
top = store.search_with_bleed(q, k=8, bleed=1)
snips, ids = store.build_context_snippets(top)

# Chat with context (RAG)
cog = CognitionChat(cfg)
cog.set_system("Answer strictly from the provided sources and cite them.")
cog.add_context(snips)
answer = cog.raw_chat(q, stream=False)
print(answer)

# Persist turn + provenance
user_mem = mem.add_memory(session, "user", q, "user_input", ["mode:knowledge_rag"])
asst_mem = mem.add_memory(session, "assistant", answer, "cog_output", ["mode:knowledge_rag"])
inv_id = mem.add_invocation(session, "knowledge_rag", q, {"k": 8, "bleed": 1})
mem.link_invocation_memory(inv_id, user_mem, "produced")
mem.link_invocation_memory(inv_id, asst_mem, "produced")
for cid in ids:
    store.link_invocation_chunk(inv_id, cid)
```
