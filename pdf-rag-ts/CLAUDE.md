# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**TypeScript server:**
```bash
npm run build   # Compile TypeScript → dist/
npm start       # Run compiled server on port 3000 (node dist/server.js)
npx ts-node src/test_rag.ts          # Run RAG smoke test directly
npx ts-node src/test_optimization.ts # Run optimization test directly
```

**Gradio UI** (requires the TS server to be running first):
```bash
cd ui
pip install -r requirements.txt
python gradio_ts_app.py              # Launches UI on http://localhost:7861
```

No linter is configured. TypeScript strict mode is enabled — `npm run build` serves as the type-check gate.

## Prerequisites

All three services must be running before starting the server:

1. **Weaviate** — vector database, run via Docker (`../docker-compose.yml` at parent directory)
2. **Ollama** — local LLM engine; models auto-pulled on first run (`nomic-embed-text`, `qwen2.5:14b`, and a rewrite model)
3. **LlamaParse** — set `LLAMAPARSE_API_KEY` env var (free key from cloud.llamaindex.ai)

## Configuration

Runtime config lives in `configs/base.yaml`. Key knobs:

- `weaviate.url`, `weaviate.collectionName` — vector DB target
- `ollama.embeddingModel` (`nomic-embed-text`), `ollama.generativeModel` (`qwen2.5:14b`), `ollama.rewriteModel`
- `chunking.chunkSize` (512), `chunking.chunkOverlap` (100), `chunking.minContentLength` (150)
- `retrieval.alpha` (0.75 = 75% semantic / 25% BM25), `retrieval.limit` (5)
- `redis.cacheUrl`, `redis.ttl` — optional response caching

Config is loaded by `src/config/config.ts` → `loadConfig()`, which reads `configs/base.yaml` from the project root.

## Architecture

The app is a local, fully-offline PDF question-answering system. The entry point is `src/server.ts`, which exposes four HTTP endpoints on port 3000 (`POST /process`, `POST /query`, `GET /stats`, `POST /reset`) and delegates all logic to `LocalRAGSystem` in `src/index.ts`.

### 5-Stage Pipeline

```
PDF → [Parser] → [Chunker] → [Importer] → [Searcher] → [Generator] → Answer
```

| Stage | Module | Key detail |
|-------|--------|-----------|
| **Parse** | `src/ingestion/parser.ts` | Calls LlamaParse API; returns `PageData[]` (markdown + page number) |
| **Chunk** | `src/chunking/chunker.ts` | Splits markdown by headers then sentences; 4 strategies: STANDARD, HIERARCHICAL, CONTEXTUAL, FULL_PAGE |
| **Index** | `src/indexing/schema.ts` + `importer.ts` | Weaviate collection with `text2vec-ollama` module; embeddings generated at insert time |
| **Search** | `src/retrieval/searcher.ts` | Hybrid search (`withHybrid`, alpha=0.75); optional page/section filters; drops chunks < 150 chars |
| **Generate** | `src/generation/generator.ts` + `src/prompting/query_augmentation.ts` | Query rewriting + HyDE before retrieval; assembled context sent to Ollama `/api/generate` (120s timeout) |

### Key Design Decisions

- **Ollama called directly** (not via Weaviate's `generate.nearText`) so the full assembled prompt is visible and controllable.
- **Embeddings live in Weaviate** via the `text2vec-ollama` vectorizer module — chunks are stored as plain text and vectors are auto-generated on import.
- **Page metadata is preserved** throughout chunking (`pageNumber`, `precedingHeaders`, `sectionPath`) to enable filtered retrieval and accurate citations.
- **Query augmentation** (`src/prompting/query_augmentation.ts`) runs query rewriting and HyDE (Hypothetical Document Embeddings) before vector search to improve recall.

### Infrastructure

- `src/infra/ollama.ts` — checks `/api/tags` on startup and auto-pulls any missing models before the server accepts requests.
- `src/utils/helpers.ts` — chunk type definitions, citation formatting (`[Doc N, p.X]`), cosine similarity, and `parallelMap`.
- `src/caching/` and `src/attribution/` directories exist as placeholders but contain no implementation yet.
