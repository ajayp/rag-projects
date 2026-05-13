import type { WeaviateClient } from 'weaviate-ts-client';

export async function setupSchema(
    client: any, // Using any for now as weaviate-ts-client types can be tricky
    embeddingModel: string,
    generativeModel: string,
    weaviateOllamaUrl: string,
    collectionName: string
): Promise<void> {
    try {
        const schemaRes = await client.schema.getter().do();
        const exists = schemaRes.classes?.some((c: any) => c.class === collectionName);

        if (exists) {
            console.log("✅ Using existing Weaviate collection");
            return;
        }

        const classObj = {
            class: collectionName,
            description: "Chunks of documents for RAG with local Ollama and page awareness",
            vectorizer: "text2vec-ollama",
            moduleConfig: {
                "text2vec-ollama": {
                    apiEndpoint: weaviateOllamaUrl,
                    model: embeddingModel,
                },
                "generative-ollama": {
                    apiEndpoint: weaviateOllamaUrl,
                    model: generativeModel,
                }
            },
            properties: [
                { name: "content", dataType: ["text"], description: "The main content of the chunk" },
                { name: "title", dataType: ["text"], description: "Document title or filename" },
                { name: "sourceFile", dataType: ["text"], description: "Original filename" },
                { name: "chunkIndex", dataType: ["int"], description: "Global index of this chunk in the document" },
                { name: "chunkSize", dataType: ["int"], description: "Size of the chunk in characters" },
                { name: "precedingHeaders", dataType: ["text[]"], description: "Headers that provide context for this chunk" },
                { name: "overlapContent", dataType: ["text"], description: "Content from adjacent chunks for context" },
                { name: "documentHash", dataType: ["text"], description: "Hash of the original document for deduplication" },
                { name: "processedAt", dataType: ["date"], description: "When this chunk was processed" },
                { name: "pageNumber", dataType: ["int"], description: "Page number where this chunk appears" },
                { name: "pageChunkIndex", dataType: ["int"], description: "Index of this chunk within its page" },
                { name: "totalPages", dataType: ["int"], description: "Total number of pages in the document" },
                { name: "chunksInPage", dataType: ["int"], description: "Total chunks in this page" },
                { name: "chunkMethod", dataType: ["text"], description: "Method used for chunking" },
                { name: "pageCharCount", dataType: ["int"], description: "Character count of the source page" },
                { name: "parentContent", dataType: ["text"], description: "The full parent section for hierarchical retrieval" },
                { name: "contextSummary", dataType: ["text"], description: "A summary of the page context" },
            ],
        };

        await client.schema.classCreator().withClass(classObj).do();
        console.log("✅ Weaviate collection created successfully with page-aware schema!");
    } catch (error) {
        console.error(`Error creating collection: ${error}`);
        throw error;
    }
}
