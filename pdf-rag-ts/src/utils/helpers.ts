import * as path from 'path';

export interface ChunkMetadata {
    score?: number;
    distance?: number;
}

export interface Chunk {
    content: string;
    title: string;
    sourceFile: string;
    chunkIndex: number;
    precedingHeaders: string[];
    pageNumber: number;
    pageChunkIndex: number;
    totalPages: number;
    parentContent?: string;
    contextSummary?: string;
    _additional?: ChunkMetadata;
}

export function objToDict(obj: any): Chunk {
    return {
        content: obj.properties.content,
        title: obj.properties.title,
        sourceFile: obj.properties.sourceFile,
        chunkIndex: obj.properties.chunkIndex,
        precedingHeaders: obj.properties.precedingHeaders || [],
        pageNumber: obj.properties.pageNumber,
        pageChunkIndex: obj.properties.pageChunkIndex,
        totalPages: obj.properties.totalPages,
        parentContent: obj.properties.parentContent,
        contextSummary: obj.properties.contextSummary,
        _additional: {
            score: obj.metadata?.score,
            distance: obj.metadata?.distance,
        },
    };
}

export function formatChunkCitation(chunk: Chunk, index: number): string {
    const headers = (chunk.precedingHeaders || []).join(' > ');
    let page = `Page ${chunk.pageNumber ?? '?'}`;
    if (chunk.totalPages) {
        page += `/${chunk.totalPages}`;
    }
    let label = `${index + 1}. ${path.basename(chunk.sourceFile)} (${page})`;
    if (headers) {
        label += ` - ${headers}`;
    }
    return label;
}

export function cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i]! * b[i]!;
        normA += a[i]! * a[i]!;
        normB += b[i]! * b[i]!;
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
}

// Splits on sentence-ending punctuation; preserves trailing whitespace as a delimiter.
// Handles common abbreviations (Mr., Dr., etc.) by requiring a capital letter after the break.
export function splitIntoSentences(text: string): string[] {
    return text
        .split(/(?<=[.!?])\s+(?=[A-Z"'‘“]|\d)/)
        .map(s => s.trim())
        .filter(s => s.length > 0);
}

/**
 * Executes an async mapper function over an array of items with a concurrency limit.
 */
export async function parallelMap<T, R>(
    items: T[],
    mapper: (item: T, index: number) => Promise<R>,
    concurrency: number
): Promise<R[]> {
    const results: R[] = new Array(items.length);
    let currentIndex = 0;

    async function worker() {
        while (currentIndex < items.length) {
            const index = currentIndex++;
            const item = items[index];
            if (item !== undefined) {
                results[index] = await mapper(item, index);
            }
        }
    }

    const workers = Array.from({ length: Math.min(concurrency, items.length) }, worker);
    await Promise.all(workers);
    return results;
}
