import * as os from 'os';
import * as path from 'path';
import weaviate from 'weaviate-ts-client';
import type { WeaviateClient } from 'weaviate-ts-client';

import { loadConfig } from './config/config.js';
import type { Config } from './config/config.js';
import { DocumentParser } from './ingestion/parser.js';
import type { PageData } from './ingestion/parser.js';
import { MarkdownChunker } from './chunking/chunker.js';
import type { ChunkData, ChunkingStrategy } from './chunking/chunker.js';
import { setupSchema } from './indexing/schema.js';
import { importToWeaviate } from './indexing/importer.js';
import { Searcher } from './retrieval/searcher.js';
import { QueryAugmenter } from './prompting/query_augmentation.js';
import { AnswerGenerator } from './generation/generator.js';
import { setupOllamaModels } from './infra/ollama.js';
import { formatChunkCitation } from './utils/helpers.js';
import type { Chunk } from './utils/helpers.js';

export interface RAGOptions {
    weaviateUrl?: string;
    ollamaUrl?: string;
    llamaParseApiKey?: string;
    embeddingModel?: string;
    generativeModel?: string;
    rewriteModel?: string;
    weaviateOllamaUrl?: string;
}

export class LocalRAGSystem {
    private config: Config;
    private ollamaUrl: string;
    private weaviateOllamaUrl: string;
    private embeddingModel: string;
    private generativeModel: string;
    private collectionName: string;
    private client: any; // WeaviateClient
    private parser: DocumentParser;
    private chunker: MarkdownChunker;
    private searcher: Searcher;
    private augmenter: QueryAugmenter;
    private generator: AnswerGenerator;
    private summaryGenerator: AnswerGenerator;

    constructor(options: RAGOptions = {}) {
        this.config = loadConfig();

        this.ollamaUrl = options.ollamaUrl || this.config.ollama.url;
        this.weaviateOllamaUrl = options.weaviateOllamaUrl || this.config.weaviate.ollamaUrl;
        this.embeddingModel = options.embeddingModel || this.config.ollama.embeddingModel;
        this.generativeModel = options.generativeModel || this.config.ollama.generativeModel;
        this.collectionName = this.config.weaviate.collectionName;

        const weaviateUrl = options.weaviateUrl || this.config.weaviate.url;
        const url = new URL(weaviateUrl);

        this.client = (weaviate as any).client({
            scheme: url.protocol.replace(':', ''),
            host: `${url.hostname}${url.port ? ':' + url.port : ''}`,
        });

        this.parser = new DocumentParser(options.llamaParseApiKey);
        this.chunker = new MarkdownChunker(this.config.chunking.chunkSize, this.config.chunking.chunkOverlap);
        this.searcher = new Searcher(this.client, this.collectionName, this.config.chunking.minContentLength);
        this.augmenter = new QueryAugmenter(this.ollamaUrl, this.generativeModel, options.rewriteModel || this.config.ollama.rewriteModel);
        this.generator = new AnswerGenerator(this.ollamaUrl, this.generativeModel);
        this.summaryGenerator = new AnswerGenerator(this.ollamaUrl, options.rewriteModel || this.config.ollama.rewriteModel);

        this.initialize();

        // Handle cleanup
        process.on('SIGINT', () => this.close());
        process.on('SIGTERM', () => this.close());
    }

    private async initialize() {
        await setupOllamaModels(this.ollamaUrl, this.embeddingModel, this.generativeModel, this.config.ollama.rewriteModel);
        await setupSchema(this.client, this.embeddingModel, this.generativeModel, this.weaviateOllamaUrl, this.collectionName);
    }

    async documentExists(filePath: string): Promise<boolean> {
        const basename = path.basename(filePath);
        const result = await this.client.graphql.get()
            .withClassName(this.collectionName)
            .withFields('sourceFile')
            .withWhere({
                path: ['sourceFile'],
                operator: 'Like',
                valueText: `*${basename}`
            })
            .withLimit(1)
            .do();
        const hits = result.data.Get[this.collectionName] || [];
        return hits.length > 0;
    }

    async processDocument(filePath: string, options: { useLlamaParse?: boolean, strategy?: ChunkingStrategy } = {}): Promise<void> {
        const { useLlamaParse = true, strategy = 'STANDARD' } = options;

        if (await this.documentExists(filePath)) {
            throw new Error(`ALREADY_EXISTS:${path.basename(filePath)}`);
        }

        console.log(`\n🚀 Processing document: ${filePath} (Strategy: ${strategy})`);

        let pagesData: PageData[];
        if (useLlamaParse) {
            console.log("1. 📄 Parsing with LlamaParse (page-aware)...");
            pagesData = await this.parser.parseWithLlamaParse(filePath);
        } else {
            console.log("1. 📄 Loading local markdown file...");
            pagesData = await this.parser.parseLocalMarkdown(filePath);
        }

        console.log("2. ✂️  Chunking with strategy: " + strategy);

        const summaryProvider = async (text: string) => {
            const prompt = `Summarize this page of a document in exactly one concise sentence. Focus on the main topic and key entities mentioned.

Page Content:
${text.substring(0, 3000)}

Summary:`;
            try {
                return await this.summaryGenerator.generate(prompt);
            } catch (e) {
                return "";
            }
        };

        const chunks = await this.chunker.chunkMarkdownWithPages(
            pagesData,
            filePath,
            strategy,
            strategy === 'CONTEXTUAL' ? summaryProvider : undefined
        );

        console.log("3. 📥 Importing to Weaviate...");
        await importToWeaviate(this.client, chunks, this.collectionName);

        const pageSummary: Record<number, number> = {};
        for (const chunk of chunks) {
            pageSummary[chunk.pageNumber] = (pageSummary[chunk.pageNumber] || 0) + 1;
        }

        console.log(`✅ Successfully processed ${filePath}:`);
        console.log(`   📄 ${pagesData.length} pages`);
        console.log(`   📝 ${chunks.length} total chunks`);
        Object.entries(pageSummary).sort(([a], [b]) => Number(a) - Number(b)).forEach(([pageNum, chunkCount]) => {
            console.log(`   📄 Page ${pageNum}: ${chunkCount} chunks`);
        });
        console.log();
    }

    async askQuestion(
        question: string,
        options: {
            maxChunks?: number;
            sectionFilter?: string[];
            pageFilter?: number[];
            sourceFile?: string;
            alpha?: number;
            useHyDE?: boolean;
            useRewrite?: boolean;
            useCache?: boolean;
        } = {}
    ): Promise<string> {
        const {
            maxChunks = 5,
            alpha = 0.75,
            useHyDE = false,
            useRewrite = false,
        } = options;

        console.log(`\n🤔 Question: ${question}`);

        let vector: number[] | undefined = undefined;
        // Always use the original question's vector to maintain the core semantic intent
        vector = await this.embed(question);

        let searchQuery = question;
        if (useRewrite) {
            // Expand the text query for BM25/Keyword search, but keep the original vector for semantic search
            const expansion = await this.augmenter.rewriteQuery(question);
            searchQuery = expansion;
        }
        if (useHyDE) {
            // HyDE usually replaces the search query entirely for the vector search, 
            // but in hybrid search, we can use the original vector + HyDE text
            searchQuery = await this.augmenter.generateHypotheticalAnswer(searchQuery);
        }

        let chunks: Chunk[];
        if (options.pageFilter) {
            chunks = await this.searcher.searchByPage(searchQuery, options.pageFilter, maxChunks, options.sourceFile, alpha, vector);
            console.log(`🔍 Found ${chunks.length} chunks in pages: ${options.pageFilter}`);
        } else if (options.sectionFilter) {
            chunks = await this.searcher.sectionFilteredSearch(searchQuery, options.sectionFilter, maxChunks, options.sourceFile, alpha, vector);
            console.log(`🔍 Found ${chunks.length} chunks in sections: ${options.sectionFilter}`);
        } else {
            chunks = await this.searcher.search(searchQuery, maxChunks, options.sourceFile, alpha, vector);
            console.log(`🔍 Found ${chunks.length} relevant chunks`);
        }

        if (chunks.length === 0) {
            return "❌ No relevant information found in the documents.";
        }

        // Deduplicate chunks
        const seen = new Set<string>();
        const uniqueChunks: Chunk[] = [];
        for (const chunk of chunks) {
            const key = `${chunk.sourceFile}-${chunk.chunkIndex}`;
            if (!seen.has(key)) {
                seen.add(key);
                uniqueChunks.push(chunk);
            }
        }
        const finalChunks = uniqueChunks;

        const contextParts: string[] = [];
        for (let i = 0; i < finalChunks.length; i++) {
            const chunk = finalChunks[i];
            if (chunk) {
                const citation = formatChunkCitation(chunk, i);
                contextParts.push(`Source ${i + 1}: [${citation}]\n${chunk.content}`);
            }
        }

        const context = "\n\n" + "=".repeat(50) + "\n\n" + contextParts.join("\n\n");

        const prompt = `Answer this question using ONLY the sources provided below: ${question}

Sources:
${context}

Instructions:
- Use ONLY information explicitly stated in the sources above. Do NOT use outside knowledge.
- Do NOT invent definitions, acronym expansions, or explanations not present in the sources.
- If the sources do not contain enough information to answer, say exactly: "The provided documents do not contain enough information to answer this question."
- Reference specific document sections and page numbers when relevant
- Keep your answer concise but complete

Answer:`;

        let generatedAnswer: string;
        try {
            generatedAnswer = await this.generator.generate(prompt);
        } catch (error) {
            console.error(`⚠️ Ollama generation failed: ${error}`);
            generatedAnswer = `I found relevant information but couldn't generate a proper answer. Here's what I found:\n\n${context.substring(0, 1000)}...`;
        }

        let sourceFooter = "\n\n📚 Sources:\n";
        for (let i = 0; i < finalChunks.length; i++) {
            const chunk = finalChunks[i];
            if (chunk) {
                sourceFooter += `  ${formatChunkCitation(chunk, i)}\n`;
            }
        }

        return generatedAnswer + sourceFooter;
    }

    async getDocumentStats(): Promise<any> {
        try {
            const result = await this.client.graphql.aggregate()
                .withClassName(this.collectionName)
                .withFields('meta { count }')
                .do();

            const totalChunks = result.data.Aggregate[this.collectionName][0].meta.count;

            const response = await this.client.graphql.get()
                .withClassName(this.collectionName)
                .withFields('sourceFile totalPages chunkMethod')
                .withLimit(totalChunks)
                .do();

            const objects = response.data.Get[this.collectionName] || [];
            const stats: any = { totalChunks, documents: {} };

            for (const obj of objects) {
                const src = obj.sourceFile || "unknown";
                const totalPages = obj.totalPages;
                if (!stats.documents[src]) {
                    stats.documents[src] = { chunks: 0 };
                }
                stats.documents[src].chunks += 1;
                if (totalPages && !stats.documents[src].pages) {
                    stats.documents[src].pages = totalPages;
                }
                if (obj.chunkMethod && !stats.documents[src].strategy) {
                    stats.documents[src].strategy = obj.chunkMethod;
                }
            }
            return stats;
        } catch (error) {
            console.error(`Error getting stats: ${error}`);
            return { totalChunks: 0, documents: {} };
        }
    }

    async reset(): Promise<void> {
        const schemaRes = await this.client.schema.getter().do();
        const exists = schemaRes.classes?.some((c: any) => c.class === this.collectionName);

        if (exists) {
            await this.client.schema.classDeleter().withClassName(this.collectionName).do();
        }

        await setupSchema(
            this.client,
            this.embeddingModel,
            this.generativeModel,
            this.weaviateOllamaUrl,
            this.collectionName
        );
        console.log("✅ Collection reset — all documents removed.");
    }

    private async embed(text: string): Promise<number[] | undefined> {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        try {
            const response = await fetch(`${this.ollamaUrl}/api/embeddings`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: this.embeddingModel,
                    prompt: text,
                }),
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                return undefined;
            }

            const data = await response.json() as { embedding?: number[] };
            return data.embedding;
        } catch (error) {
            clearTimeout(timeoutId);
            return undefined;
        }
    }

    close(): void {
        // weaviate-ts-client v2 doesn't have an explicit close()
    }
}
