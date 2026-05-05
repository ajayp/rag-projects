# Local Document Q&A with Weaviate, Ollama & LlamaParse

## Overview
A fully local RAG system that answers natural language questions over PDF documents. Parses and chunks PDFs using **LlamaParse**, stores embeddings in **Weaviate**, and generates cited answers with page references using **Ollama** — no external LLM API required.

## Features
- Intelligent PDF parsing with **LlamaParse** — preserves page structure, tables, and sections
- Vector storage and retrieval with **Weaviate**
- Local embedding and generation with **Ollama** (`nomic-embed-text` + `qwen2.5:14b`)
- Hybrid search — BM25 keyword + semantic vector with a tunable alpha slider
- Two-pass chunking: markdown structure first, then sentence-level size limits — chunks never cross page boundaries
- **Result filtering** — bare header chunks and table-of-contents entries are filtered out before being passed to the LLM, preventing low-content results from occupying source slots
- **Result deduplication** — identical chunks are deduplicated before context assembly
- Answers include **source citations** — page numbers and section names from the original document
- **Query expansion** — runs via `gemma3:1b` (fast, lightweight) to add synonyms and related terms before searching, bridging vocabulary gaps between your question and the document's language
- **HyDE** (Hypothetical Document Embeddings) — LLM generates a hypothetical answer passage and searches with that, improving retrieval for conceptual questions
- **Semantic cache** — repeated or semantically similar questions are served from Redis without hitting the LLM again, reducing latency on common queries
- Browser UI with drag-and-drop PDF upload and document filter

---

## Query Pipeline

```
┌─────────────────────────────────┐
│         User Question           │
│          (Gradio UI)            │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       Semantic Cache Check      │◄─── SemanticCache (src/caching/semantic_cache.py)
└────────────────┬────────────────┘
        hit │         │ miss
            │         ▼
            │  ┌──────────────────┐
            │  │  Query Rewrite?  │  (optional)
            │  └────────┬─────────┘
            │    no │       │ yes
            │       │       ▼
            │       │  ┌───────────────────────┐
            │       │  │  rewrite_query()       │
            │       │  │  gemma3:1b (fast)      │
            │       │  └──────────┬────────────┘
            │       │             │
            │       └──────┬──────┘
            │              │
            │              ▼
            │  ┌──────────────────────┐
            │  │      HyDE?           │  (optional)
            │  └──────────┬───────────┘
            │    no │         │ yes
            │       │         ▼
            │       │  ┌────────────────────────┐
            │       │  │ generate_hypothetical_ │
            │       │  │ answer() qwen2.5:14b   │
            │       │  └──────────┬─────────────┘
            │       │             │
            │       └──────┬──────┘
            │              │
            │              ▼
            │  ┌───────────────────────────────┐
            │  │          search()             │
            │  │  Hybrid BM25 + Vector (α=0.75)│
            │  │  Weaviate ← nomic-embed-text  │
            │  │                               │
            │  │  variants:                    │
            │  │  • search_by_page()           │
            │  │  • section_filtered_search()  │
            │  └──────────────┬────────────────┘
            │                 │
            │                 ▼
            │  ┌───────────────────────────────┐
            │  │  Filter & Deduplicate Chunks  │
            │  │  (drop chunks < 150 chars)    │
            │  └──────────────┬────────────────┘
            │                 │
            │                 ▼
            │  ┌───────────────────────────────┐
            │  │  Build Cited Context (top-K)  │
            │  └──────────────┬────────────────┘
            │                 │
            │                 ▼
            │  ┌───────────────────────────────┐
            │  │       ask_question()          │
            │  │       qwen2.5:14b             │
            │  └──────────────┬────────────────┘
            │                 │
            │                 ▼
            │  ┌───────────────────────────────┐
            │  │      Store in Cache           │
            │  └──────────────┬────────────────┘
            │                 │
            └────────►────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Answer + Citations   │
         │      (Gradio UI)       │
         └────────────────────────┘
```

---

<img width="1434" height="984" alt="image" src="https://github.com/user-attachments/assets/58590644-e719-4cac-b9e7-95908231ab16" />


---

## Prerequisites

### 1. Docker
Required to run Weaviate. [Docker](https://docker.com) or [OrbStack](https://orbstack.dev) (faster, recommended).

### 2. Ollama
Runs the embedding and generative models locally. Download from [ollama.com](https://ollama.com). The app will pull the required models automatically if not already installed.

### 3. LlamaParse API Key
Used for intelligent PDF parsing. Get a free key from [Llama Cloud](https://cloud.llamaindex.ai), then set it as an environment variable:
```bash
export LLAMAPARSE_API_KEY="your_api_key_here"
```

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Create a Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Weaviate:**
   ```bash
   docker compose up -d
   ```

---

## Usage

```bash
python -m apps.gradio_app
# or: python main.py
```

Opens at `http://localhost:7860`. Upload PDFs via the sidebar, then ask questions in the chat.

### Search options

| Option | What it *actually* does | When it's the right tool | When it will absolutely betray you |
| --- | --- | --- | --- |
| **Query Expansion** | Deterministically broadens the query using known synonyms, aliases, taxonomies, or controlled vocabularies. | When user phrasing ≠ corpus phrasing. When you need higher recall without semantic drift. | When expansions are LLM-generated instead of dictionary-driven — you get invented synonyms and polluted retrieval. |
| **HyDE** | Generates a hypothetical "ideal answer", embeds it, and retrieves documents semantically similar to that hallucinated answer. | Conceptual, prose-heavy, narrative corpora. Questions about mechanisms, explanations, or high-level reasoning. | Technical domains, API names, model names, framework questions, entity-sensitive retrieval. HyDE hallucinates terms → retrieval collapses. |
| **Search mode slider** | Blends BM25 keyword matching with semantic vector search. 0 = keyword only, 1 = semantic only. | Default 0.75 works well for technical docs. Slide toward 0 for exact-term queries, toward 1 for conceptual ones. | Pure BM25 (0) misses synonyms; pure vector (1) misses rare technical terms that don't embed distinctively. |

### Not implemented: reranking

Cross-encoder reranking (e.g. `ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-large`) would improve retrieval precision further but is not implemented. It requires `sentence-transformers` as an additional dependency and adds latency on top of local LLM generation. With hybrid search and query expansion already in place, reranking is unlikely to be the bottleneck for single-document use cases — but is a natural next step for multi-document setups where candidates are harder to rank.

### Not implemented: multi-hop retrieval

Multi-hop retrieval (iterative retrieval where the answer to one query informs the next) is not implemented. For single-document Q&A, most questions are single-hop and don't require chaining lookups. It becomes worthwhile when answers genuinely span multiple large documents or require following references across sections.

---

## Cleanup

```bash
docker compose down           # stop Weaviate + Redis
ollama rm nomic-embed-text    # remove embedding model (optional)
ollama rm qwen2.5:14b         # remove generative model (optional)
ollama rm gemma3:1b           # remove query rewrite model (optional)
```


<img width="1373" height="381" alt="image" src="https://github.com/user-attachments/assets/62419464-9f10-4a3d-aacd-aa4c1a82ce65" />
