import redis as _redis_module
import requests
from typing import Optional, List

from redisvl.extensions.cache.llm import SemanticCache as _RedisvlCache
from redisvl.utils.vectorize import CustomVectorizer


def _make_vectorizer(ollama_url: str, model: str) -> CustomVectorizer:
    def embed(text: str) -> List[float]:
        r = requests.post(
            f"{ollama_url}/api/embed",
            json={"model": model, "input": text},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        vecs = data.get("embeddings") or data.get("embedding")
        if not vecs:
            raise ValueError(f"Unexpected Ollama embed response: {data}")
        return vecs[0] if isinstance(vecs[0], list) else vecs

    return CustomVectorizer(
        embed=embed,
        embed_many=lambda texts: [embed(t) for t in texts],
    )


class SemanticCache:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        distance_threshold: float = 0.12,
        ttl: int = 3600,
    ):
        self._redis = _redis_module.from_url(redis_url)
        self._redis_url = redis_url
        self._ollama_url = ollama_url
        self._embedding_model = embedding_model
        self._distance_threshold = distance_threshold
        self._ttl = ttl
        self._cache: Optional[_RedisvlCache] = None  # lazy — Ollama must be up first

    def _get_cache(self) -> _RedisvlCache:
        if self._cache is None:
            self._cache = _RedisvlCache(
                name="rag_cache",
                vectorizer=_make_vectorizer(self._ollama_url, self._embedding_model),
                redis_url=self._redis_url,
                distance_threshold=self._distance_threshold,
                ttl=self._ttl,
            )
        return self._cache

    def is_available(self) -> bool:
        try:
            self._redis.ping()
            return True
        except Exception:
            return False

    def get(self, question: str, source_file: Optional[str] = None, settings: str = "") -> Optional[str]:
        try:
            hits = self._get_cache().check(prompt=question, num_results=10)
            for hit in hits:
                meta = hit.get("metadata", {})
                if (
                    meta.get("source_file", "") == (source_file or "")
                    and meta.get("settings", "") == settings
                ):
                    dist = hit.get("vector_distance")
                    dist_str = f"{dist:.3f}" if isinstance(dist, (int, float)) else "?"
                    print(f"[cache] hit — distance {dist_str}")
                    return f"⚡ *Cached response (similarity distance: {dist_str})*\n\n{hit['response']}"
        except Exception as e:
            print(f"[cache] get error (skipping): {e}")
        return None

    def set(self, question: str, answer: str, source_file: Optional[str] = None, settings: str = "") -> None:
        try:
            self._get_cache().store(
                prompt=question,
                response=answer,
                metadata={"source_file": source_file or "", "settings": settings},
            )
        except Exception as e:
            print(f"[cache] set error (skipping): {e}")

    def clear(self) -> bool:
        if not self.is_available():
            return False
        try:
            self._get_cache().clear()
            return True
        except Exception as e:
            print(f"[cache] clear error: {e}")
            return False

    def stats(self) -> dict:
        if not self.is_available():
            return {"available": False}
        return {"available": True}
