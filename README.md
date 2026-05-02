# RAG Projects

A collection of Retrieval-Augmented Generation (RAG) projects built with [Weaviate](https://weaviate.io/) and [Ollama](https://ollama.com/) for fully local or hybrid vector search and LLM inference.

## Projects

### [pdf-rag](./pdf-rag/)
**Local Document Q&A — Python**

A production-style RAG system for answering natural language questions over PDF documents. Parses PDFs with LlamaParse (preserving page and section structure), stores embeddings in Weaviate, and generates cited answers with page references using Ollama — no external LLM API required at inference time.

Features hybrid BM25 + semantic search, query expansion, and HyDE (Hypothetical Document Embeddings) to improve retrieval across different query types.

**Stack:** Python · Weaviate · Ollama · LlamaParse · Browser UI

---

### [rag-tutorial](./rag-tutorial/)
**RAG Tutorial — Node.js / TypeScript**

A step-by-step tutorial that builds a RAG pipeline from scratch using a 7k-book dataset. Covers collection setup, data ingestion with automatic vectorization, semantic (vector) search, and generative (RAG) search — good starting point for understanding the difference between the two.

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
```

See each project's README for setup and run instructions.
