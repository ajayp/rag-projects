import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
import requests
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core import Document
from llama_cloud_services import LlamaParse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import hashlib
import json
import os
from urllib.parse import urlparse

class LocalRAGSystem:
    def __init__(self, 
                 weaviate_url: str = "http://localhost:8080",
                 ollama_url: str = "http://localhost:11434",
                 llamaparse_api_key: str = None,
                 embedding_model: str = "nomic-embed-text",
                 generative_model: str = "qwen2.5:14b"):
        """
        Complete local RAG system with Ollama and Weaviate v4
        """
        self.weaviate_url = weaviate_url
        self.ollama_url = ollama_url
        self.llamaparse_api_key = llamaparse_api_key
        if llamaparse_api_key is None:
            self.llamaparse_api_key = os.getenv('LLAMAPARSE_API_KEY')
        self.embedding_model = embedding_model
        self.generative_model = generative_model
        
        parsed = urlparse(self.weaviate_url)
        self.client = weaviate.connect_to_local(host=parsed.hostname, port=parsed.port or 8080)
        
        # Initialize chunking strategy
        self.markdown_parser = MarkdownNodeParser()
        self.sentence_splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=100,
            separator=" ",
            paragraph_separator="\n\n"
        )
        
        self.setup_ollama_models()
        self.setup_schema()
    
    def setup_ollama_models(self):
        """
        Ensure required Ollama models are available
        """
        print("Setting up Ollama models...")
        
        # Check if models are available
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                
                # Pull embedding model if not available
                if not any(self.embedding_model in model for model in models):
                    print(f"Pulling embedding model: {self.embedding_model}")
                    self._pull_ollama_model(self.embedding_model)
                
                # Pull generative model if not available
                if not any(self.generative_model in model for model in models):
                    print(f"Pulling generative model: {self.generative_model}")
                    self._pull_ollama_model(self.generative_model)
                
                print("✅ Ollama models ready!")
            else:
                print("⚠️  Could not connect to Ollama. Make sure it's running on port 11434")
        except Exception as e:
            print(f"⚠️  Ollama setup error: {e}")
    
    def _pull_ollama_model(self, model_name: str):
        """Pull a model in Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model_name},
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'status' in data:
                        print(f"  {data['status']}")
                    if data.get('status') == 'success':
                        break
        except Exception as e:
            print(f"Error pulling model {model_name}: {e}")
    
    def setup_schema(self):
        """
        Create Weaviate schema with enhanced metadata for page-aware processing
        """
        collection_name = "DocumentChunk"

        if self.client.collections.exists(collection_name):
            print("✅ Using existing Weaviate collection")
            return

        try:
            collection = self.client.collections.create(
                name=collection_name,
                description="Chunks of documents for RAG with local Ollama and page awareness",
                vector_config=Configure.Vectorizer.text2vec_ollama(
                    api_endpoint=self.ollama_url,
                    model=self.embedding_model
                ),
                generative_config=Configure.Generative.ollama(
                    api_endpoint=self.ollama_url,
                    model=self.generative_model,
                ),
                properties=[
                    Property(name="content", data_type=DataType.TEXT, description="The main content of the chunk"),
                    Property(name="title", data_type=DataType.TEXT, description="Document title or filename"),
                    Property(name="sourceFile", data_type=DataType.TEXT, description="Original filename"),
                    Property(name="chunkIndex", data_type=DataType.INT, description="Global index of this chunk in the document"),
                    Property(name="chunkSize", data_type=DataType.INT, description="Size of the chunk in characters"),
                    Property(name="precedingHeaders", data_type=DataType.TEXT_ARRAY, description="Headers that provide context for this chunk"),
                    Property(name="overlapContent", data_type=DataType.TEXT, description="Content from adjacent chunks for context"),
                    Property(name="documentHash", data_type=DataType.TEXT, description="Hash of the original document for deduplication"),
                    Property(name="processedAt", data_type=DataType.DATE, description="When this chunk was processed"),
                    Property(name="pageNumber", data_type=DataType.INT, description="Page number where this chunk appears"),
                    Property(name="pageChunkIndex", data_type=DataType.INT, description="Index of this chunk within its page"),
                    Property(name="totalPages", data_type=DataType.INT, description="Total number of pages in the document"),
                    Property(name="chunksInPage", data_type=DataType.INT, description="Total chunks in this page"),
                    Property(name="chunkMethod", data_type=DataType.TEXT, description="Method used for chunking"),
                    Property(name="pageCharCount", data_type=DataType.INT, description="Character count of the source page")
                ]
            )
            print("✅ Weaviate collection created successfully with page-aware schema!")
            
        except Exception as e:
            print(f"Error creating collection: {e}")
            try:
                collection = self.client.collections.get(collection_name)
                print("Using existing collection")
            except Exception as e2:
                print(f"Failed to get existing collection: {e2}")
                raise

    def parse_with_llamaparse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse document with LlamaParse and return page-aware structured data
        """
        if not self.llamaparse_api_key:
            raise ValueError("LlamaParse API key required for document parsing")

        parser = LlamaParse(
            api_key=self.llamaparse_api_key, 
            num_workers=1, 
            check_interval=5, 
            verbose=True,
            split_by_page=True
        )
        
        result = parser.parse(file_path)

        if not result:
            raise Exception("Failed to parse document or received empty response.")

        documents = result.get_markdown_documents(split_by_page=True)
        
        if not documents:
            raise Exception("No documents returned from LlamaParse")
        
        print(f"📄 LlamaParse returned {len(documents)} pages")

        pages_data = []
        for i, doc in enumerate(documents):
            page_content = doc.text.strip()
            if page_content:  # Only process non-empty pages
                pages_data.append({
                    "content": page_content,
                    "page_number": i + 1,
                    "source_file": file_path,
                    "total_pages": len(documents)
                })
                print(f"  📄 Page {i + 1}: {len(page_content)} characters")
        
        print(f"✅ Processed {len(pages_data)} non-empty pages")
        return pages_data
    
    def parse_local_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load markdown file directly and treat as single page
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return [{
            "content": content,
            "page_number": 1,
            "source_file": file_path,
            "total_pages": 1
        }]
    
    def extract_headers_context(self, content: str, chunk_text: str) -> List[str]:
        """
        Extract relevant headers that provide context for a chunk
        """
        lines = content.split('\n')
        chunk_start = content.find(chunk_text[:100])  # Find approximate position
        
        headers = []
        current_headers = []
        
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                header_text = line.lstrip('# ').strip()
                
                # Keep only headers at this level or above
                current_headers = [h for h in current_headers if h[0] < level]
                current_headers.append((level, header_text))
                
                # If we've passed the chunk position, use current context
                line_pos = sum(len(l) + 1 for l in lines[:i])
                if chunk_start >= 0 and line_pos >= chunk_start:
                    headers = [h[1] for h in current_headers]
                    break
        
        # If no headers found yet, use the current context
        if not headers and current_headers:
            headers = [h[1] for h in current_headers]
        
        return headers
    
    def chunk_markdown_with_pages(self, pages_data: List[Dict[str, Any]], source_file: str) -> List[Dict[str, Any]]:
        """
        Enhanced chunking that preserves page information
        """
        print(f"Chunking content from {len(pages_data)} pages...")
        
        all_chunks = []
        doc_hash = hashlib.md5("".join([p["content"] for p in pages_data]).encode()).hexdigest()
        
        for page_data in pages_data:
            page_content = page_data["content"]
            page_num = page_data["page_number"]
            total_pages = page_data["total_pages"]
            
            print(f"  Processing page {page_num}/{total_pages}...")
            
            # Create document for this page
            doc = Document(text=page_content)
            
            # Parse with markdown-aware parser first
            markdown_nodes = self.markdown_parser.get_nodes_from_documents([doc])
            
            # Then apply sentence splitting
            page_chunks = []
            for node in markdown_nodes:
                sentence_nodes = self.sentence_splitter.get_nodes_from_documents([Document(text=node.text)])
                page_chunks.extend(sentence_nodes)
            
            # Convert to our format with page metadata
            for i, chunk in enumerate(page_chunks):
                # Extract header context for this chunk
                headers = self.extract_headers_context(page_content, chunk.text)
                
                # Get overlap content (within page and cross-page)
                overlap_content = ""
                if i > 0:
                    overlap_content += page_chunks[i-1].text[-100:]
                elif page_num > 1 and all_chunks:  # First chunk of page, get from previous page
                    overlap_content += all_chunks[-1]["content"][-100:]
                
                if i < len(page_chunks) - 1:
                    overlap_content += page_chunks[i+1].text[:100]
                
                chunk_data = {
                    "content": chunk.text,
                    "title": os.path.splitext(os.path.basename(source_file))[0],
                    "sourceFile": source_file,
                    "chunkIndex": len(all_chunks),
                    "chunkSize": len(chunk.text),
                    "precedingHeaders": headers,
                    "overlapContent": overlap_content,
                    "documentHash": doc_hash,
                    "processedAt": datetime.now(timezone.utc),
                    "pageNumber": page_num,
                    "pageChunkIndex": i,
                    "totalPages": total_pages,
                    "chunksInPage": len(page_chunks),
                    "chunkMethod": "page_aware_markdown_sentence_split",
                    "pageCharCount": len(page_content),
                }
                all_chunks.append(chunk_data)
            
            print(f"    ✅ Page {page_num}: {len(page_chunks)} chunks")
        
        print(f"✅ Created {len(all_chunks)} total chunks from {len(pages_data)} pages")
        return all_chunks
    
    def import_to_weaviate(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Import chunks to Weaviate with batch processing using v4 API
        """
        print(f"Importing {len(chunks)} chunks to Weaviate...")
        
        collection = self.client.collections.get("DocumentChunk")
        
        # Use v4 batch insert
        try:
            with collection.batch.dynamic() as batch:
                for i, chunk in enumerate(chunks):
                    batch.add_object(
                        properties=chunk
                    )
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i+1}/{len(chunks)} chunks")
        
            print("✅ Import completed!")
            
        except Exception as e:
            print(f"Error during import: {e}")
            # Try individual imports as fallback
            print("Trying individual imports...")
            for i, chunk in enumerate(chunks):
                try:
                    collection.data.insert(chunk)
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i+1}/{len(chunks)} chunks")
                except Exception as chunk_error:
                    print(f"Error inserting chunk {i}: {chunk_error}")
    
    def process_document(self, file_path: str, use_llamaparse: bool = True) -> None:
        """
        Full pipeline: Parse -> Chunk -> Import (Page-Aware)
        """
        print(f"\n🚀 Processing document: {file_path}")
        
        # Step 1: Parse document into pages
        if use_llamaparse and self.llamaparse_api_key:
            print("1. 📄 Parsing with LlamaParse (page-aware)...")
            pages_data = self.parse_with_llamaparse(file_path)
        else:
            print("1. 📄 Loading local markdown file...")
            pages_data = self.parse_local_markdown(file_path)
        
        # Step 2: Chunk with page awareness
        print("2. ✂️  Chunking with page awareness...")
        chunks = self.chunk_markdown_with_pages(pages_data, file_path)
        
        # Step 3: Import to Weaviate
        print("3. 📥 Importing to Weaviate...")
        self.import_to_weaviate(chunks)
        
        # Show summary
        page_summary = {}
        for chunk in chunks:
            page_num = chunk["pageNumber"]
            if page_num not in page_summary:
                page_summary[page_num] = 0
            page_summary[page_num] += 1
        
        print(f"✅ Successfully processed {file_path}:")
        print(f"   📄 {len(pages_data)} pages")
        print(f"   📝 {len(chunks)} total chunks")
        for page_num, chunk_count in sorted(page_summary.items()):
            print(f"   📄 Page {page_num}: {chunk_count} chunks")
        print()
    
    def _obj_to_dict(self, obj) -> Dict:
        return {
            "content": obj.properties.get("content"),
            "title": obj.properties.get("title"),
            "sourceFile": obj.properties.get("sourceFile"),
            "chunkIndex": obj.properties.get("chunkIndex"),
            "precedingHeaders": obj.properties.get("precedingHeaders", []),
            "pageNumber": obj.properties.get("pageNumber"),
            "pageChunkIndex": obj.properties.get("pageChunkIndex"),
            "totalPages": obj.properties.get("totalPages"),
            "_additional": {
                "score": getattr(obj.metadata, "score", None),
                "distance": getattr(obj.metadata, "distance", None),
            },
        }

    def _source_filter(self, source_file: Optional[str] = None):
        return Filter.by_property("sourceFile").equal(source_file) if source_file else None

    def search(self, query: str, limit: int = 5, source_file: Optional[str] = None, min_content_length: int = 150, alpha: float = 0.75) -> List[Dict]:
        collection = self.client.collections.get("DocumentChunk")
        # alpha: 0.0 = pure BM25 (keyword), 1.0 = pure vector (semantic)
        # Over-fetch so we have candidates left after filtering bare headers.
        response = collection.query.hybrid(
            query=query,
            alpha=alpha,
            limit=limit * 3,
            filters=self._source_filter(source_file),
            return_metadata=wvc.query.MetadataQuery(score=True)
        )
        results = []
        for obj in response.objects:
            content = obj.properties.get("content", "")
            if len(content) >= min_content_length:
                results.append(self._obj_to_dict(obj))
                if len(results) >= limit:
                    break
        # Fall back to unfiltered results if nothing passes the length threshold
        if not results:
            results = [self._obj_to_dict(obj) for obj in response.objects[:limit]]
        return results

    def section_filtered_search(self, query: str, required_sections: List[str], limit: int = 5, source_file: Optional[str] = None, alpha: float = 0.75) -> List[Dict]:
        """Search within specific document sections."""
        collection = self.client.collections.get("DocumentChunk")
        response = collection.query.hybrid(
            query=query,
            alpha=alpha,
            limit=limit * 3,
            filters=self._source_filter(source_file),
            return_metadata=wvc.query.MetadataQuery(score=True)
        )
        results = []
        for obj in response.objects:
            headers = obj.properties.get("precedingHeaders", [])
            if any(s.lower() in h.lower() for s in required_sections for h in headers):
                results.append(self._obj_to_dict(obj))
                if len(results) >= limit:
                    break
        return results

    def search_by_page(self, query: str, page_numbers: Optional[List[int]] = None, limit: int = 5, source_file: Optional[str] = None, alpha: float = 0.75) -> List[Dict]:
        """Search within specific pages."""
        collection = self.client.collections.get("DocumentChunk")
        response = collection.query.hybrid(
            query=query,
            alpha=alpha,
            limit=limit * 3,
            filters=self._source_filter(source_file),
            return_metadata=wvc.query.MetadataQuery(score=True)
        )
        if not page_numbers:
            return [self._obj_to_dict(obj) for obj in response.objects[:limit]]
        results = []
        for obj in response.objects:
            if obj.properties.get("pageNumber") in page_numbers:
                results.append(self._obj_to_dict(obj))
                if len(results) >= limit:
                    break
        return results

    def generate_hypothetical_answer(self, question: str) -> str:
        prompt = f"""Write a short, technically detailed passage (2-4 sentences) that directly answers the following question. Write as if you are the document being searched — use specific terms, model names, and technical details that would appear in a technical document.

Question: {question}
Passage:"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.generative_model, "prompt": prompt, "stream": False},
                timeout=30,
            )
            response.raise_for_status()
            hypothetical = response.json().get("response", question).strip()
            print(f"💭 HyDE passage: {hypothetical[:100]}...")
            return hypothetical
        except Exception:
            return question

    def ask_question(self, question: str, max_chunks: int = 5, section_filter: Optional[List[str]] = None, page_filter: Optional[List[int]] = None, source_file: Optional[str] = None, alpha: float = 0.75, use_hyde: bool = False) -> str:
        print(f"\n🤔 Question: {question}")

        search_query = question
        if use_hyde:
            search_query = self.generate_hypothetical_answer(question)

        if page_filter:
            chunks = self.search_by_page(search_query, page_filter, max_chunks, source_file, alpha=alpha)
            print(f"🔍 Found {len(chunks)} chunks in pages: {page_filter}")
        elif section_filter:
            chunks = self.section_filtered_search(search_query, section_filter, max_chunks, source_file, alpha=alpha)
            print(f"🔍 Found {len(chunks)} chunks in sections: {section_filter}")
        else:
            chunks = self.search(search_query, max_chunks, source_file, alpha=alpha)
            print(f"🔍 Found {len(chunks)} relevant chunks")
        
        if not chunks:
            return "❌ No relevant information found in the documents."

        # Deduplicate by chunkIndex to avoid repeating the same chunk in context
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            key = (chunk.get("sourceFile"), chunk.get("chunkIndex"))
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)
        chunks = unique_chunks

        # Build context with page and header information
        context_parts = []
        for i, chunk in enumerate(chunks):
            headers = " > ".join(chunk.get('precedingHeaders', []))
            page_info = f"Page {chunk.get('pageNumber', '?')}"
            if chunk.get('totalPages'):
                page_info += f"/{chunk.get('totalPages')}"
            
            source_info = f"[{os.path.basename(chunk['sourceFile'])} - {page_info}"
            if headers:
                source_info += f" - {headers}"
            source_info += "]"
            
            context_parts.append(f"Source {i+1}: {source_info}\n{chunk['content']}")
        
        context = "\n\n" + "="*50 + "\n\n".join(context_parts)
        
        # Build the prompt
        prompt = f"""Answer this question using ONLY the sources provided below: {question}

    Sources:
    {context}

    Instructions:
    - Use ONLY information explicitly stated in the sources above. Do NOT use outside knowledge.
    - Do NOT invent definitions, acronym expansions, or explanations not present in the sources.
    - If the sources do not contain enough information to answer, say exactly: "The provided documents do not contain enough information to answer this question."
    - Reference specific document sections and page numbers when relevant
    - Keep your answer concise but complete

    Answer:"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.generative_model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            response.raise_for_status()
            generated_answer = response.json().get("response", "").strip()
        except Exception as e:
            print(f"⚠️ Ollama generation failed: {e}")
            generated_answer = f"I found relevant information but couldn't generate a proper answer. Here's what I found:\n\n{context[:1000]}..."
        
        # Add source information with page numbers
        source_info = "\n\n📚 Sources:\n"
        for i, chunk in enumerate(chunks):
            headers = " > ".join(chunk.get('precedingHeaders', []))
            page_info = f"Page {chunk.get('pageNumber', '?')}"
            if chunk.get('totalPages'):
                page_info += f"/{chunk.get('totalPages')}"
            
            source_line = f"  {i+1}. {os.path.basename(chunk['sourceFile'])} ({page_info})"
            if headers:
                source_line += f" - {headers}"
            source_info += source_line + "\n"
        
        return generated_answer + source_info

    def get_document_stats(self) -> Dict:
        """
        Get statistics about imported documents with page information - Simple version
        """
        collection = self.client.collections.get("DocumentChunk")
        
        try:
            # Get total count
            total_response = collection.aggregate.over_all(total_count=True)
            total_chunks = total_response.total_count
            
            # Simple approach: fetch all chunks and group manually
            response = collection.query.fetch_objects(
                limit=total_chunks,
                return_properties=["sourceFile", "totalPages"]
            )
            
            stats = {
                "total_chunks": total_chunks,
                "documents": {}
            }
            
            # Group by source file manually
            for obj in response.objects:
                source_file = obj.properties.get("sourceFile", "unknown")
                total_pages = obj.properties.get("totalPages")
                
                if source_file not in stats["documents"]:
                    stats["documents"][source_file] = {"chunks": 0}
                
                stats["documents"][source_file]["chunks"] += 1
                
                # Set pages if we haven't already
                if total_pages and "pages" not in stats["documents"][source_file]:
                    stats["documents"][source_file]["pages"] = total_pages
            
            return stats
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                "total_chunks": 0,
                "documents": {}
            }
    
    def rewrite_query(self, question: str) -> str:
        prompt = f"""You are a search query expander for a technical document retrieval system. Expand the question into a short search query that includes the core topic plus relevant synonyms and related technical terms. Return only the expanded query, nothing else.

Question: what models to use for rerank
Expanded: rerank cross-encoder reranker bi-encoder two-stage retrieval ms-marco bge-reranker

Question: how does chunking work
Expanded: chunking text splitting document segmentation sentence splitting chunk size overlap

Question: what is RAG
Expanded: RAG retrieval augmented generation retrieval-augmented pipeline architecture

Question: how to reduce hallucination
Expanded: hallucination reduction grounding faithfulness factuality citation verification

Question: {question}
Expanded:"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.generative_model, "prompt": prompt, "stream": False},
                timeout=30,
            )
            response.raise_for_status()
            rewritten = response.json().get("response", question).strip()
            print(f"🔄 Query expanded: '{question}' → '{rewritten}'")
            return rewritten
        except Exception:
            return question

    def reset(self):
        collection_name = "DocumentChunk"
        if self.client.collections.exists(collection_name):
            self.client.collections.delete(collection_name)
        self.setup_schema()
        print("✅ Collection reset — all documents removed.")

    def close(self):
        self.client.close()

