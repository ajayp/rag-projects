import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import * as yaml from 'js-yaml';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export interface Config {
    weaviate: {
        url: string;
        ollamaUrl: string;
        collectionName: string;
    };
    ollama: {
        url: string;
        embeddingModel: string;
        generativeModel: string;
        rewriteModel: string;
    };
    chunking: {
        chunkSize: number;
        chunkOverlap: number;
        minContentLength: number;
    };
    retrieval: {
        defaultAlpha: number;
        defaultLimit: number;
    };
    cache: {
        redisUrl: string;
        distanceThreshold: number;
        ttl: number;
    };
}

export function loadConfig(configPath?: string): Config {
    const defaultPath = path.join(__dirname, '..', '..', 'configs', 'base.yaml');
    const actualPath = configPath || defaultPath;

    const fileContents = fs.readFileSync(actualPath, 'utf8');
    const raw = yaml.load(fileContents) as any;

    return {
        weaviate: {
            url: raw.weaviate.url,
            ollamaUrl: raw.weaviate.ollama_url,
            collectionName: raw.weaviate.collection_name,
        },
        ollama: {
            url: raw.ollama.url,
            embeddingModel: raw.ollama.embedding_model,
            generativeModel: raw.ollama.generative_model,
            rewriteModel: raw.ollama.rewrite_model,
        },
        chunking: {
            chunkSize: raw.chunking.chunk_size,
            chunkOverlap: raw.chunking.chunk_overlap,
            minContentLength: raw.chunking.min_content_length,
        },
        retrieval: {
            defaultAlpha: raw.retrieval.default_alpha,
            defaultLimit: raw.retrieval.default_limit,
        },
        cache: {
            redisUrl: raw.cache.redis_url,
            distanceThreshold: raw.cache.distance_threshold,
            ttl: raw.cache.ttl,
        },
    };
}
