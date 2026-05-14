from typing import List, Dict, Optional, Any

import weaviate.classes as wvc

from utils.helpers import obj_to_dict, source_filter


class Searcher:
    def __init__(self, client, collection_name: str, min_content_length: int = 150):
        self.client = client
        self.collection_name = collection_name
        self.min_content_length = min_content_length

    def _collection(self):
        return self.client.collections.get(self.collection_name)

    def search(
        self,
        query: str,
        limit: int = 5,
        source_file: Optional[str] = None,
        alpha: float = 0.75,
        vector: Optional[List[float]] = None,
    ) -> List[Dict]:
        kwargs: Dict[str, Any] = dict(
            query=query,
            alpha=alpha,
            limit=limit * 3,
            filters=source_filter(source_file),
            return_metadata=wvc.query.MetadataQuery(score=True),
        )
        if vector is not None:
            kwargs["vector"] = vector
        response = self._collection().query.hybrid(**kwargs)
        results = []
        for obj in response.objects:
            content = obj.properties.get("content", "")
            if len(content) >= self.min_content_length:
                results.append(obj_to_dict(obj))
                if len(results) >= limit:
                    break
        if not results:
            results = [obj_to_dict(obj) for obj in response.objects[:limit]]
        return results

    def section_filtered_search(
        self,
        query: str,
        required_sections: List[str],
        limit: int = 5,
        source_file: Optional[str] = None,
        alpha: float = 0.75,
        vector: Optional[List[float]] = None,
    ) -> List[Dict]:
        kwargs: Dict[str, Any] = dict(
            query=query,
            alpha=alpha,
            limit=limit * 3,
            filters=source_filter(source_file),
            return_metadata=wvc.query.MetadataQuery(score=True),
        )
        if vector is not None:
            kwargs["vector"] = vector
        response = self._collection().query.hybrid(**kwargs)
        results = []
        for obj in response.objects:
            headers = obj.properties.get("precedingHeaders", [])
            if any(s.lower() in h.lower() for s in required_sections for h in headers):
                results.append(obj_to_dict(obj))
                if len(results) >= limit:
                    break
        return results

    def search_by_page(
        self,
        query: str,
        page_numbers: Optional[List[int]] = None,
        limit: int = 5,
        source_file: Optional[str] = None,
        alpha: float = 0.75,
        vector: Optional[List[float]] = None,
    ) -> List[Dict]:
        kwargs: Dict[str, Any] = dict(
            query=query,
            alpha=alpha,
            limit=limit * 3,
            filters=source_filter(source_file),
            return_metadata=wvc.query.MetadataQuery(score=True),
        )
        if vector is not None:
            kwargs["vector"] = vector
        response = self._collection().query.hybrid(**kwargs)
        if not page_numbers:
            return [obj_to_dict(obj) for obj in response.objects[:limit]]
        results = []
        for obj in response.objects:
            if obj.properties.get("pageNumber") in page_numbers:
                results.append(obj_to_dict(obj))
                if len(results) >= limit:
                    break
        return results
