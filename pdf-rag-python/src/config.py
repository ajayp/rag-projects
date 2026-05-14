from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    weaviate_url: str
    weaviate_ollama_url: str
    collection_name: str
    ollama_url: str
    embedding_model: str
    generative_model: str
    rewrite_model: str
    chunk_size: int
    chunk_overlap: int
    min_content_length: int
    default_alpha: float
    default_limit: int
    redis_url: str
    cache_distance_threshold: float
    cache_ttl: int


def load_config(path: Path = None) -> Config:
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "base.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(
        weaviate_url=raw["weaviate"]["url"],
        weaviate_ollama_url=raw["weaviate"]["ollama_url"],
        collection_name=raw["weaviate"]["collection_name"],
        ollama_url=raw["ollama"]["url"],
        embedding_model=raw["ollama"]["embedding_model"],
        generative_model=raw["ollama"]["generative_model"],
        rewrite_model=raw["ollama"]["rewrite_model"],
        chunk_size=raw["chunking"]["chunk_size"],
        chunk_overlap=raw["chunking"]["chunk_overlap"],
        min_content_length=raw["chunking"]["min_content_length"],
        default_alpha=raw["retrieval"]["default_alpha"],
        default_limit=raw["retrieval"]["default_limit"],
        redis_url=raw["cache"]["redis_url"],
        cache_distance_threshold=raw["cache"]["distance_threshold"],
        cache_ttl=raw["cache"]["ttl"],
    )
