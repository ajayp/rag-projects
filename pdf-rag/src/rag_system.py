import atexit
import os
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import requests

import weaviate

from src.config import load_config
from src.ingestion.parser import DocumentParser
from src.chunking.chunker import MarkdownChunker
from src.indexing.schema import setup_schema
from src.indexing.importer import import_to_weaviate
from src.retrieval.searcher import Searcher
from src.prompting.query_augmentation import QueryAugmenter
from src.generation.generator import AnswerGenerator
from src.infra.ollama import setup_ollama_models
from src.caching.semantic_cache import SemanticCache
from utils.helpers import format_chunk_citation


class LocalRAGSystem:
    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        ollama_url: str = "http://localhost:11434",
        llamaparse_api_key: str = None,
        embedding_model: str = "nomic-embed-text",
        generative_model: str = "qwen2.5:14b",
        rewrite_model: str = "gemma3:1b",
        weaviate_ollama_url: str = "http://host.docker.internal:11434",
        cache: Optional[SemanticCache] = None,
    ):
        cfg = load_config()

        self.ollama_url = ollama_url
        self.weaviate_ollama_url = weaviate_ollama_url
        self.embedding_model = embedding_model
        self.generative_model = generative_model
        self.collection_name = cfg.collection_name
        self.cache = cache

        parsed = urlparse(weaviate_url)
        self.client = weaviate.connect_to_local(host=parsed.hostname, port=parsed.port or 8080)

        self.parser = DocumentParser(
            llamaparse_api_key=llamaparse_api_key or os.getenv("LLAMAPARSE_API_KEY")
        )
        self.chunker = MarkdownChunker(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        self.searcher = Searcher(
            client=self.client,
            collection_name=self.collection_name,
            min_content_length=cfg.min_content_length,
        )
        self.augmenter = QueryAugmenter(
            ollama_url=ollama_url,
            generative_model=generative_model,
            rewrite_model=rewrite_model,
        )
        self.generator = AnswerGenerator(
            ollama_url=ollama_url,
            generative_model=generative_model,
        )

        setup_ollama_models(ollama_url, embedding_model, generative_model, rewrite_model)
        setup_schema(self.client, embedding_model, generative_model, weaviate_ollama_url, self.collection_name)
        atexit.register(self.close)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def process_document(self, file_path: str, use_llamaparse: bool = True) -> None:
        print(f"\n🚀 Processing document: {file_path}")

        if self.cache:
            if self.cache.clear():
                print(f"🗑️  Cache cleared (re-indexing {os.path.basename(file_path)})")

        if use_llamaparse and self.parser.llamaparse_api_key:
            print("1. 📄 Parsing with LlamaParse (page-aware)...")
            pages_data = self.parser.parse_with_llamaparse(file_path)
        else:
            print("1. 📄 Loading local markdown file...")
            pages_data = self.parser.parse_local_markdown(file_path)

        print("2. ✂️  Chunking with page awareness...")
        chunks = self.chunker.chunk_markdown_with_pages(pages_data, file_path)

        print("3. 📥 Importing to Weaviate...")
        import_to_weaviate(self.client, chunks, self.collection_name)

        page_summary: Dict[int, int] = {}
        for chunk in chunks:
            page_num = chunk["pageNumber"]
            page_summary[page_num] = page_summary.get(page_num, 0) + 1

        print(f"✅ Successfully processed {file_path}:")
        print(f"   📄 {len(pages_data)} pages")
        print(f"   📝 {len(chunks)} total chunks")
        for page_num, chunk_count in sorted(page_summary.items()):
            print(f"   📄 Page {page_num}: {chunk_count} chunks")
        print()

    def ask_question(
        self,
        question: str,
        max_chunks: int = 5,
        section_filter: Optional[List[str]] = None,
        page_filter: Optional[List[int]] = None,
        source_file: Optional[str] = None,
        alpha: float = 0.75,
        use_hyde: bool = False,
        use_rewrite: bool = False,
        use_cache: bool = True,
    ) -> str:
        print(f"\n🤔 Question: {question}")
        settings = f"expand={use_rewrite}|hyde={use_hyde}|alpha={alpha}"

        if use_cache and self.cache:
            cached = self.cache.get(question, source_file, settings)
            if cached is not None:
                return cached

        search_query = question
        vector = None
        if use_rewrite:
            # Embed the original question BEFORE rewriting so vector search stays
            # anchored to the user's intent; the expanded text only affects BM25.
            vector = self._embed(question)
            search_query = self.augmenter.rewrite_query(search_query)
        if use_hyde:
            search_query = self.augmenter.generate_hypothetical_answer(search_query)

        if page_filter:
            chunks = self.searcher.search_by_page(search_query, page_filter, max_chunks, source_file, alpha=alpha, vector=vector)
            print(f"🔍 Found {len(chunks)} chunks in pages: {page_filter}")
        elif section_filter:
            chunks = self.searcher.section_filtered_search(search_query, section_filter, max_chunks, source_file, alpha=alpha, vector=vector)
            print(f"🔍 Found {len(chunks)} chunks in sections: {section_filter}")
        else:
            chunks = self.searcher.search(search_query, max_chunks, source_file, alpha=alpha, vector=vector)
            print(f"🔍 Found {len(chunks)} relevant chunks")

        if not chunks:
            return "❌ No relevant information found in the documents."

        seen: set = set()
        unique_chunks = []
        for chunk in chunks:
            key = (chunk.get("sourceFile"), chunk.get("chunkIndex"))
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)
        chunks = unique_chunks

        context_parts = []
        for i, chunk in enumerate(chunks):
            citation = format_chunk_citation(chunk, i)
            context_parts.append(f"Source {i+1}: [{citation}]\n{chunk['content']}")

        context = "\n\n" + "=" * 50 + "\n\n".join(context_parts)

        prompt = (
            f"Answer this question using ONLY the sources provided below: {question}\n\n"
            f"Sources:\n{context}\n\n"
            "Instructions:\n"
            "- Use ONLY information explicitly stated in the sources above. Do NOT use outside knowledge.\n"
            "- Do NOT invent definitions, acronym expansions, or explanations not present in the sources.\n"
            '- If the sources do not contain enough information to answer, say exactly: "The provided documents do not contain enough information to answer this question."\n'
            "- Reference specific document sections and page numbers when relevant\n"
            "- Keep your answer concise but complete\n\n"
            "Answer:"
        )

        try:
            generated_answer = self.generator.generate(prompt)
        except Exception as e:
            print(f"⚠️ Ollama generation failed: {e}")
            generated_answer = f"I found relevant information but couldn't generate a proper answer. Here's what I found:\n\n{context[:1000]}..."

        source_footer = "\n\n📚 Sources:\n"
        for i, chunk in enumerate(chunks):
            source_footer += f"  {format_chunk_citation(chunk, i)}\n"

        full_answer = generated_answer + source_footer
        if use_cache and self.cache:
            self.cache.set(question, full_answer, source_file, settings)
        return full_answer

    def get_document_stats(self) -> Dict[str, Any]:
        collection = self.client.collections.get(self.collection_name)
        try:
            total_response = collection.aggregate.over_all(total_count=True)
            total_chunks = total_response.total_count

            response = collection.query.fetch_objects(
                limit=total_chunks,
                return_properties=["sourceFile", "totalPages"],
            )

            stats: Dict[str, Any] = {"total_chunks": total_chunks, "documents": {}}
            for obj in response.objects:
                src = obj.properties.get("sourceFile", "unknown")
                total_pages = obj.properties.get("totalPages")
                if src not in stats["documents"]:
                    stats["documents"][src] = {"chunks": 0}
                stats["documents"][src]["chunks"] += 1
                if total_pages and "pages" not in stats["documents"][src]:
                    stats["documents"][src]["pages"] = total_pages
            return stats
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_chunks": 0, "documents": {}}

    def reset(self) -> None:
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)
        setup_schema(
            self.client,
            self.embedding_model,
            self.generator.generative_model,
            self.weaviate_ollama_url,
            self.collection_name,
        )
        if self.cache:
            self.cache.clear()
        print("✅ Collection reset — all documents removed.")

    def _embed(self, text: str) -> Optional[List[float]]:
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("embedding")
        except Exception:
            return None

    def close(self) -> None:
        self.client.close()
