import { vectorizer, generative } from 'weaviate-client';
import { withWeaviateClient, COLLECTION_NAME } from './utils/utils.ts';

const OLLAMA_ENDPOINT = 'http://host.docker.internal:11434';

withWeaviateClient(async (client) => {
    await client.collections.delete(COLLECTION_NAME);
    await client.collections.create({
        name: COLLECTION_NAME,
        vectorizers: vectorizer.text2VecOllama({
            apiEndpoint: OLLAMA_ENDPOINT,
            model: 'nomic-embed-text',
        }),
        generative: generative.ollama({
            apiEndpoint: OLLAMA_ENDPOINT,
            model: 'llama3.2',
        }),
    });
    console.log(`Collection ${COLLECTION_NAME} created with Ollama vectorizer and generative model.`);
}).catch(console.error);
