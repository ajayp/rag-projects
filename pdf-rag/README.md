# Local Document Q&A with Weaviate, Ollama & LlamaParse

## Overview
A fully local RAG system that answers natural language questions over PDF documents. Parses and chunks PDFs using **LlamaParse**, stores embeddings in **Weaviate**, and generates cited answers with page references using **Ollama** — no external LLM API required.

## Features
- Intelligent PDF parsing with **LlamaParse** — preserves page structure, tables, and sections
- Vector storage and retrieval with **Weaviate**
- Local embedding and generation with **Ollama** (`nomic-embed-text` + `qwen2.5:14b`)
- Hybrid search — BM25 keyword + semantic vector with a tunable alpha slider
- Two-pass chunking: markdown structure first, then sentence-level size limits — chunks never cross page boundaries
- Answers include **source citations** — page numbers and section names from the original document
- **Query expansion** — LLM adds synonyms and related terms before searching, bridging vocabulary gaps between your question and the document's language
- **HyDE** (Hypothetical Document Embeddings) — LLM generates a hypothetical answer passage and searches with that, improving retrieval for conceptual questions
- Browser UI with drag-and-drop PDF upload and document filter

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
python app.py
```

Opens at `http://localhost:7860`. Upload PDFs via the sidebar, then ask questions in the chat.

### Search options

| Option | What it does | When to use |
|--------|-------------|-------------|
| **Query expansion** | Adds synonyms and related terms to your query | Your wording differs from the document's terminology (e.g. "rerank" vs "cross-encoder") |
| **HyDE** | Generates a hypothetical answer and searches with it | Conceptual questions where phrasing is the gap, not specific terms — ⚠️ the LLM may hallucinate model names or frameworks it doesn't know, leading to worse retrieval than query expansion for technical documents |
| **Search mode slider** | 0 = keyword only (BM25), 1 = semantic only (vector) | Tune per document type; default 0.75 works well for technical docs |

### Not implemented: reranking

Cross-encoder reranking (e.g. `ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-large`) would improve retrieval precision further but is not implemented. It requires `sentence-transformers` as an additional dependency and adds latency on top of local LLM generation. With hybrid search and query expansion already in place, reranking is unlikely to be the bottleneck for single-document use cases — but is a natural next step for multi-document setups where candidates are harder to rank.

---

## Cleanup

```bash
docker compose down           # stop Weaviate
ollama rm nomic-embed-text    # remove embedding model (optional)
ollama rm qwen2.5:14b         # remove generative model (optional)
```
