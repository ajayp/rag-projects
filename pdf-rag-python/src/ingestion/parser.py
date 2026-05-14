import os
from typing import List, Dict, Any

from llama_cloud_services import LlamaParse


class DocumentParser:
    def __init__(self, llamaparse_api_key: str = None):
        self.llamaparse_api_key = llamaparse_api_key or os.getenv("LLAMAPARSE_API_KEY")

    def parse_with_llamaparse(self, file_path: str) -> List[Dict[str, Any]]:
        if not self.llamaparse_api_key:
            raise ValueError("LlamaParse API key required for document parsing")

        parser = LlamaParse(
            api_key=self.llamaparse_api_key,
            num_workers=1,
            check_interval=5,
            verbose=True,
            split_by_page=True,
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
            if page_content:
                pages_data.append({
                    "content": page_content,
                    "page_number": i + 1,
                    "source_file": file_path,
                    "total_pages": len(documents),
                })
                print(f"  📄 Page {i + 1}: {len(page_content)} characters")

        print(f"✅ Processed {len(pages_data)} non-empty pages")
        return pages_data

    def parse_local_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [{
            "content": content,
            "page_number": 1,
            "source_file": file_path,
            "total_pages": 1,
        }]
