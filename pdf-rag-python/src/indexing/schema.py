from weaviate.classes.config import Configure, Property, DataType


def setup_schema(
    client,
    embedding_model: str,
    generative_model: str,
    weaviate_ollama_url: str,
    collection_name: str,
) -> None:
    if client.collections.exists(collection_name):
        print("✅ Using existing Weaviate collection")
        return

    try:
        client.collections.create(
            name=collection_name,
            description="Chunks of documents for RAG with local Ollama and page awareness",
            vectorizer_config=Configure.Vectorizer.text2vec_ollama(
                api_endpoint=weaviate_ollama_url,
                model=embedding_model,
            ),
            generative_config=Configure.Generative.ollama(
                api_endpoint=weaviate_ollama_url,
                model=generative_model,
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
                Property(name="pageCharCount", data_type=DataType.INT, description="Character count of the source page"),
            ],
        )
        print("✅ Weaviate collection created successfully with page-aware schema!")
    except Exception as e:
        print(f"Error creating collection: {e}")
        try:
            client.collections.get(collection_name)
            print("Using existing collection")
        except Exception as e2:
            print(f"Failed to get existing collection: {e2}")
            raise
