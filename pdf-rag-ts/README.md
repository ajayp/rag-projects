# Local Document Q&A with Weaviate, Ollama & LlamaParse

A fully local RAG system that answers natural language questions over PDF documents. Parses and chunks PDFs using **LlamaParse**, stores embeddings in **Weaviate**, and generates cited answers with page references using **Ollama** — no external LLM API required.

## Features
- Document parsing with LlamaParse
- Page-aware chunking with MarkdownNodeParser and SentenceSplitter
- Multiple chunking strategies, refer to **Chunking strategies** below 
- Hybrid search (BM25 + Vector)
- Query augmentation (Rewrite + HyDE)
- Answer generation with Ollama
- Natural Language Inference (NLI) for retrieval conflict detection - coming soon
- Semantic caching - coming soon

<img width="1120" height="540" alt="image" src="https://github.com/user-attachments/assets/d5cd7b16-c7d8-4840-8759-5ad54177e617" />

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


## Setup

1. **Start Ollama:**
   ```bash
   ollama serve
   ```

2. **Start Weaviate:**
   ```bash
   docker compose up -d
   ```

3. **Install Node dependencies and build:**
   ```bash
   npm install
   npm run build
   ```

4. **Start the backend server (port 3000):**
   ```bash
   npm start
   ```

5. **Set up the Gradio UI** — in a separate terminal:
   ```bash
   cd ui
   python3 -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   python gradio_ts_app.py
   ```

   Opens at `http://localhost:7861`.


## Usage

```typescript
import { LocalRAGSystem } from './dist/index.js';

const rag = new LocalRAGSystem({
    llamaParseApiKey: "your-api-key"
});

// Process a document
await rag.processDocument("path/to/document.pdf");

// Ask a question
const answer = await rag.askQuestion("What is the main topic?");
console.log(answer);
```


---

## How It Works

The pipeline runs in five stages:

**Parse → Chunk → Index → Search → Generate**

1. **Parse** — PDFs are sent to LlamaParse, which returns structured markdown split by page.
2. **Chunk** — `src/chunking/chunker.ts` splits markdown by headers then by sentence (512-char window, 100-char overlap), preserving page boundaries and section hierarchy as metadata.
3. **Index** — Chunks are batch-imported into Weaviate. The `text2vec-ollama` module (using `nomic-embed-text`) generates embeddings automatically at insert time — chunks are stored as raw text, not pre-computed vectors.
4. **Search** — `src/retrieval/searcher.ts` uses Weaviate's `withHybrid` operator, blending BM25 keyword matching with vector similarity at `alpha=0.75` (75% vector, 25% BM25). Short chunks (< 150 chars, e.g. bare headers) are filtered out before results are returned.
5. **Generate** — Retrieved chunks are assembled into a cited prompt and sent to Ollama (`qwen2.5:14b`) directly via `src/generation/generator.ts`.

### Why Ollama directly instead of Weaviate's `generate.nearText`?

Weaviate's generative search (`generate.nearText`) is designed to call an external LLM provider (OpenAI, Cohere, Anthropic, etc.) configured in the schema. This project is **fully local** — no external LLM API calls during inference — so that path isn't applicable.

Calling Ollama directly also gives full control over:
- Prompt structure and citation formatting
- Query augmentation (rewrite + HyDE) that runs *before* retrieval in `src/prompting/query_augmentation.ts`
- Streaming, temperature, and stop token configuration

---

## Configuration

### Chunking strategies

Four strategies are supported, selectable per document at ingest time:

| Strategy | Description | Best for |
|---|---|---|
| `STANDARD` (default) | 2-pass: markdown header split → 512-char sentence chunks (100 overlap) | Most structured docs |
| `HIERARCHICAL` | Smaller chunks (200 chars, 50 overlap) preserving header structure | Dense technical docs |
| `CONTEXTUAL` | `STANDARD` + headers and page summary prepended to each chunk | Ambiguous passages where retrieval needs more context |
| `FULL_PAGE` | Each markdown section kept as one chunk, no further splitting | Short sections or when full context matters |

Select a strategy when processing a document:

```typescript
await rag.processDocument("path/to/doc.pdf", { strategy: 'HIERARCHICAL' });
```

Default is `STANDARD`. The chosen strategy is stored as `chunkMethod` metadata on each chunk.

See `src/chunking/chunker.ts` for implementation details.

### Not implemented: Natural Language Inference

Natural Language Inference (NLI) model for retrieval conflict detection, scoring document pairs to identify contradictions before generation so the LLM can explicitly address disagreements rather than blending conflicting evidence into a hallucinated answer.

### Not implemented: reranking

Cross-encoder reranking (e.g. `ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-large`) would improve retrieval precision further but is not implemented. It requires `sentence-transformers` as an additional dependency and adds latency on top of local LLM generation. With hybrid search and query expansion already in place, reranking is unlikely to be the bottleneck for single-document use cases — but is a natural next step for multi-document setups where candidates are harder to rank.

### Not implemented: multi-hop retrieval

Multi-hop retrieval (iterative retrieval where the answer to one query informs the next) is not implemented. For single-document Q&A, most questions are single-hop and don't require chaining lookups. It becomes worthwhile when answers genuinely span multiple large documents or require following references across sections.



