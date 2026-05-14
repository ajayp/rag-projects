import os
from typing import Dict, Optional
from weaviate.classes.query import Filter


def obj_to_dict(obj) -> Dict:
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


def source_filter(source_file: Optional[str] = None):
    return Filter.by_property("sourceFile").equal(source_file) if source_file else None


def format_chunk_citation(chunk: Dict, index: int) -> str:
    headers = " > ".join(chunk.get("precedingHeaders", []))
    page = f"Page {chunk.get('pageNumber', '?')}"
    if chunk.get("totalPages"):
        page += f"/{chunk['totalPages']}"
    label = f"{index + 1}. {os.path.basename(chunk['sourceFile'])} ({page})"
    if headers:
        label += f" - {headers}"
    return label
