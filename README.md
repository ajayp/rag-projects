# RAG Projects

A collection of Retrieval-Augmented Generation (RAG) projects built with [Weaviate](https://weaviate.io/) and [Ollama](https://ollama.com/) for fully local or hybrid vector search and LLM inference.

## System Architecture & Philosophy
The core message behind these projects is that **robust RAG is not simply a "vector DB plus a prompt"**. Instead, I treat RAG as a layered, full-stack decision system consisting of:

*   **Ingestion:** Documents are normalized, structures are parsed, metadata is attached, and every source is versioned to ensure a traceable "chain of custody".
*   **Indexing:** I generate embeddings, optionally build lexical indexes for **BM25**, and utilize a **parent-child storage strategy** to preserve broader context while maintaining retrieval precision.
*   **Retrieval & Reranking:** At query time, candidates are retrieved using **hybrid search** (BM25 + vector). Where precision matters, a **reranker** can be layered in — though this adds latency and is not always necessary.
*   **Generation & Assembly:** The best evidence is assembled into a **bounded prompt**, instructing the model to answer *only* from the retrieved context.
*   **Evaluation & Monitoring:** Every stage should be instrumented for quality (faithfulness, relevance) to catch silent failures early — this is often the last thing built and the first thing that pays off in production.

By following this layered approach, I mitigate the issues that cause most production RAG failures: **bad evidence, weak retrieval, or poor uncertainty handling**.

## Design Philosophy: Framework Selection
For these projects, I leaned on **LlamaIndex** and **LlamaParse** because they strike a good balance between development speed, flexibility, and manageable complexity. They’re great for getting prototypes quickly, and they remove a lot of the routine wiring that doesn’t add real value. Frameworks should simplify the plumbing, not obscure the parts that matter.

If a use case pushes beyond generic framework behaviour whether due to stability, latency pressure, or customization needs, my strategy is to transition to a minimal custom orchestration layer. That keeps retrieval logic, prompt construction, evaluation, and observability transparent and easy to reason, even as the underlying framework evolves.

## Projects

### [rag-tutorial](./rag-tutorial/)
**RAG Tutorial — Node.js / TypeScript**

A step-by-step tutorial that builds a RAG pipeline from scratch using a 7k-book dataset. Covers collection setup, data ingestion with automatic vectorization, semantic (_vector_) search, and generative (_RAG_) search — good starting point for understanding the difference between the two.

**Stack:** Node.js · TypeScript · Weaviate · Ollama

---

### [pdf-rag-python](./pdf-rag-python/)
**Local Document Q&A — Python**

A production-style RAG system for answering natural language questions over PDF documents. Parses PDFs with LlamaParse (_preserving page and section structure_), stores embeddings in Weaviate, and generates cited answers with page references using Ollama — no external LLM API required at inference time.

Features hybrid BM25 + semantic search, query expansion (_via `gemma3:1b`_), HyDE (_Hypothetical Document Embeddings_), and a Redis-backed semantic cache that serves repeated or semantically similar questions without hitting the LLM again.

**Stack:** Python · Weaviate · Ollama · LlamaParse · Redis · Browser UI

---

### [pdf-rag-ts](./pdf-rag-ts/)
**Local Document Q&A — TypeScript**

A TypeScript port of the same production-style PDF Q&A system. Parses PDFs with LlamaParse, stores embeddings in Weaviate, and generates cited answers via Ollama. Supports four chunking strategies (_Standard, Hierarchical, Contextual, Full-Page_), hybrid BM25 + semantic search, and query augmentation (_rewrite + HyDE_).

**Stack:** Node.js · TypeScript · Weaviate · Ollama · LlamaParse · Browser UI

---

## Shared Prerequisites

All projects require:

- **[Docker](https://www.docker.com/)** — runs Weaviate locally
- **[Ollama](https://ollama.com/)** — runs embedding and generation models locally

```bash
# Pull the models used across projects
ollama pull nomic-embed-text   # embeddings
ollama pull qwen2.5:14b        # text generation
ollama pull gemma3:1b          # query rewrite (pdf-rag)
```

**pdf-rag-python** also requires **Redis** (_semantic cache_) — run alongside Weaviate via `docker compose up -d`.

See each project's README for setup and run instructions.
