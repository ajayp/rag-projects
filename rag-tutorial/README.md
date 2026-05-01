# RAG Tutorial: Semantic & Generative Search with Weaviate & Ollama

## Overview
A step-by-step tutorial that builds a **Retrieval-Augmented Generation (RAG)** pipeline from scratch using **Weaviate**, **Ollama**, and **Node.js**. Uses a book dataset to teach the difference between semantic search and generative search (RAG).

## What You'll Learn
- How to set up **Weaviate** locally with Docker
- How to generate **vector embeddings** with Ollama (`nomic-embed-text`)
- The difference between **semantic search** and **generative (RAG) search**
- How to use an **LLM** (`llama3.2`) to synthesize answers from retrieved documents

---
<img width="1573" alt="image" src="https://github.com/user-attachments/assets/de312e84-414b-45dd-9bed-e660d43fb9c1" />

---

## Environment Setup

### 1. Start Weaviate with Docker
```bash
docker-compose -f docker-compose.yml up -d
```
> The current `docker-compose.yml` includes Ollama modules only. For a different vectorizer, see the [Weaviate Docker Installation Guide](https://weaviate.io/developers/weaviate/installation/docker-compose).

### 2. Install Ollama and Pull Models
Download Ollama from [ollama.com](https://ollama.com), then pull the required models:
```bash
ollama pull nomic-embed-text:latest  # embeddings
ollama pull llama3.2:latest          # inference
```

## Project Setup
```bash
npm i
```

---

## Tutorial Steps

### Step 1 — Create Collection
```bash
node 1-createCollection.ts
```
Sets up a `Book` collection in Weaviate, configured with Ollama for embedding and generation.

### Step 2 — Populate Collection
```bash
node 2-populateCollection.ts
```
Loads book data from a Kaggle dataset into Weaviate.

### Step 3 — Semantic Search
```bash
node 3-semanticSearch.ts
```
Queries Weaviate using vector embeddings to find books conceptually similar to a given prompt.

### Step 4 — Generative Search (RAG)
```bash
node 4-generativeSearch.ts
```
Retrieves relevant books via semantic search, then passes them to `llama3.2` to generate a human-like response.

---

## Semantic vs. Generative Search

### Semantic Search
- Understands **meaning and context**, not just keywords
- Returns **existing records** ranked by vector similarity
- Fast, but doesn't generate new answers

### Generative Search (RAG)
- Retrieves relevant documents via semantic search
- Passes them to an **LLM** to synthesize a response
- More powerful, but slower and dependent on retrieval quality

---

## Choosing an Embedding Model

**`nomic-embed-text`** is used here because it's optimized for natural language — book titles, summaries, and reviews. Models like `snowflake-arctic-embed` are tuned for structured enterprise data (logs, financial docs) and are less effective at capturing literary and thematic meaning.

---
<img width="1504" alt="image" src="https://github.com/user-attachments/assets/a6545717-1c40-4b4f-901f-8412d20d9d60" />
