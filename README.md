# RAG Projects

A collection of Retrieval-Augmented Generation (RAG) projects built with [Weaviate](https://weaviate.io/) and [Ollama](https://ollama.com/) for fully local or hybrid vector search and LLM inference.

## Projects

### [pdf-rag](./pdf-rag/)
**Local Document Q&A — Python**

A production-style RAG system for answering natural language questions over PDF documents. Parses PDFs with LlamaParse (_preserving page and section structure_), stores embeddings in Weaviate, and generates cited answers with page references using Ollama — no external LLM API required at inference time.

Features hybrid BM25 + semantic search, query expansion (_via `gemma3:1b`_), HyDE (_Hypothetical Document Embeddings_), and a Redis-backed semantic cache that serves repeated or semantically similar questions without hitting the LLM again.

**Stack:** Python · Weaviate · Ollama · LlamaParse · Redis · Browser UI

---

### [rag-tutorial](./rag-tutorial/)
**RAG Tutorial — Node.js / TypeScript**

A step-by-step tutorial that builds a RAG pipeline from scratch using a 7k-book dataset. Covers collection setup, data ingestion with automatic vectorization, semantic (_vector_) search, and generative (_RAG_) search — good starting point for understanding the difference between the two.

**Stack:** Node.js · TypeScript · Weaviate · Ollama

---

## Shared Prerequisites

Both projects require:

- **[Docker](https://www.docker.com/)** — runs Weaviate locally
- **[Ollama](https://ollama.com/)** — runs embedding and generation models locally

```bash
# Pull the models used across projects
ollama pull nomic-embed-text   # embeddings
ollama pull qwen2.5:14b        # text generation
ollama pull gemma3:1b          # query rewrite (pdf-rag)
```

**pdf-rag** also requires **Redis** (_semantic cache_) — run alongside Weaviate via `docker compose up -d`.

See each project's README for setup and run instructions.
