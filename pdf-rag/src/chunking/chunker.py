import hashlib
import os
from datetime import datetime, timezone
from typing import List, Dict, Any

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core import Document


class MarkdownChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        self.markdown_parser = MarkdownNodeParser()
        self.sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
        )

    def extract_headers_context(self, content: str, chunk_text: str) -> List[str]:
        lines = content.split("\n")
        chunk_start = content.find(chunk_text[:100])

        headers = []
        current_headers = []

        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                header_text = line.lstrip("# ").strip()

                current_headers = [h for h in current_headers if h[0] < level]
                current_headers.append((level, header_text))

                line_pos = sum(len(l) + 1 for l in lines[:i])
                if chunk_start >= 0 and line_pos >= chunk_start:
                    headers = [h[1] for h in current_headers]
                    break

        if not headers and current_headers:
            headers = [h[1] for h in current_headers]

        return headers

    def chunk_markdown_with_pages(self, pages_data: List[Dict[str, Any]], source_file: str) -> List[Dict[str, Any]]:
        print(f"Chunking content from {len(pages_data)} pages...")

        all_chunks = []
        doc_hash = hashlib.md5("".join([p["content"] for p in pages_data]).encode()).hexdigest()

        for page_data in pages_data:
            page_content = page_data["content"]
            page_num = page_data["page_number"]
            total_pages = page_data["total_pages"]

            print(f"  Processing page {page_num}/{total_pages}...")

            doc = Document(text=page_content)
            markdown_nodes = self.markdown_parser.get_nodes_from_documents([doc])

            page_chunks = []
            for node in markdown_nodes:
                sentence_nodes = self.sentence_splitter.get_nodes_from_documents([Document(text=node.text)])
                page_chunks.extend(sentence_nodes)

            for i, chunk in enumerate(page_chunks):
                headers = self.extract_headers_context(page_content, chunk.text)

                overlap_content = ""
                if i > 0:
                    overlap_content += page_chunks[i - 1].text[-100:]
                elif page_num > 1 and all_chunks:
                    overlap_content += all_chunks[-1]["content"][-100:]

                if i < len(page_chunks) - 1:
                    overlap_content += page_chunks[i + 1].text[:100]

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
