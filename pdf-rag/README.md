# Local Document Q&A with Weaviate, Ollama & LlamaParse

## Overview
A fully local RAG system that answers natural language questions over PDF documents. Parses and chunks PDFs using **LlamaParse**, stores embeddings in **Weaviate**, and generates cited answers with page references using **Ollama** — no external LLM API required.

## Features
- Intelligent PDF parsing with **LlamaParse** — preserves page structure, tables, and sections
- Vector storage and retrieval with **Weaviate**
- Local embedding and generation with **Ollama** (`nomic-embed-text` + `llama3.2`)
- Two-pass chunking: markdown structure first, then sentence-level size limits — chunks never cross page boundaries
- Answers include **source citations** — page numbers and section names from the original document
- Browser UI with drag-and-drop PDF upload, document filter, and optional query rewriting

---
<img width="1207" alt="image" src="https://github.com/user-attachments/assets/f619be37-3802-4042-90eb-03ad2d39a544" />

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

### Browser UI (recommended)
```bash
python app.py
```
Opens at `http://localhost:7860`. Upload PDFs via the sidebar, select a document from the dropdown to scope queries, and enable **Query rewriting** to handle conversational questions.

### CLI
```bash
python cli.py
```
Interactive terminal session. Supports `page:1,2` and `section:Definitions` prefixes to filter results. Type `quit` to exit.

---

## Cleanup

```bash
docker compose down          # stop Weaviate
ollama rm nomic-embed-text   # remove embedding model (optional)
ollama rm llama3.2           # remove generative model (optional)
```
