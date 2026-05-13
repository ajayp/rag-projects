import * as crypto from 'crypto';
import * as path from 'path';
import { Document, MarkdownNodeParser, SentenceSplitter } from 'llamaindex';
import type { PageData } from '../ingestion/parser.js';
import { parallelMap } from '../utils/helpers.js';

export type ChunkingStrategy = 'STANDARD' | 'HIERARCHICAL' | 'CONTEXTUAL' | 'FULL_PAGE';

export interface ChunkData {
    content: string;
    title: string;
    sourceFile: string;
    chunkIndex: number;
    chunkSize: number;
    precedingHeaders: string[];
    overlapContent: string;
    documentHash: string;
    processedAt: string;
    pageNumber: number;
    pageChunkIndex: number;
    totalPages: number;
    chunksInPage: number;
    chunkMethod: string;
    pageCharCount: number;
    parentContent?: string | undefined; // For HIERARCHICAL
    contextSummary?: string | undefined; // For CONTEXTUAL
}

export class MarkdownChunker {
    private markdownParser: MarkdownNodeParser;
    private standardSplitter: SentenceSplitter;
    private hierarchicalSplitter: SentenceSplitter;

    constructor(chunkSize: number = 512, chunkOverlap: number = 100) {
        this.markdownParser = new MarkdownNodeParser();
        this.standardSplitter = new SentenceSplitter({
            chunkSize: chunkSize,
            chunkOverlap: chunkOverlap,
        });
        this.hierarchicalSplitter = new SentenceSplitter({
            chunkSize: 200, // Small chunks for embedding
            chunkOverlap: 50,
        });
    }

    private extractHeadersContext(pageContent: string, chunkText: string, preParsedHeaders?: { level: number, text: string, pos: number }[]): string[] {
        const chunkStart = pageContent.indexOf(chunkText.substring(0, 100));
        if (chunkStart < 0) return [];

        let headers: { level: number, text: string, pos: number }[] = [];
        
        if (preParsedHeaders) {
            headers = preParsedHeaders;
        } else {
            const lines = pageContent.split('\n');
            let currentPos = 0;
            for (const line of lines) {
                if (line.trim().startsWith('#')) {
                    const match = line.match(/^(#+)\s*(.*)$/);
                    if (match && match[1] && match[2]) {
                        headers.push({ 
                            level: match[1].length, 
                            text: match[2].trim(),
                            pos: currentPos
                        });
                    }
                }
                currentPos += line.length + 1;
            }
        }

        const relevantHeaders: string[] = [];
        let currentLevel = 999;

        for (let i = headers.length - 1; i >= 0; i--) {
            const header = headers[i];
            if (header && header.pos <= chunkStart && header.level < currentLevel) {
                relevantHeaders.unshift(header.text);
                currentLevel = header.level;
                if (currentLevel === 1) break;
            }
        }

        return relevantHeaders;
    }

    async chunkMarkdownWithPages(
        pagesData: PageData[], 
        sourceFile: string, 
        strategy: ChunkingStrategy = 'STANDARD',
        summaryProvider?: (text: string) => Promise<string>
    ): Promise<ChunkData[]> {
        console.log(`Chunking content from ${pagesData.length} pages using strategy: ${strategy}...`);

        const allChunks: ChunkData[] = [];
        const fullContent = pagesData.map(p => p.content).join('');
        const docHash = crypto.createHash('md5').update(fullContent).digest('hex');

        // PRE-FETCH SUMMARIES IN PARALLEL (Concurrency limit 4)
        const pageSummaries = new Map<number, string>();
        if (strategy === 'CONTEXTUAL' && summaryProvider) {
            console.log(`  Fetching contextual summaries for ${pagesData.length} pages in parallel...`);
            await parallelMap(pagesData, async (page) => {
                const summary = await summaryProvider(page.content);
                pageSummaries.set(page.pageNumber, summary);
            }, 4);
            console.log(`  ✅ All summaries generated.`);
        }

        for (const pageData of pagesData) {
            const pageContent = pageData.content;
            const pageNum = pageData.pageNumber;
            const totalPages = pageData.totalPages;
            const pageSummary = pageSummaries.get(pageNum) || "";

            console.log(`  Processing page ${pageNum}/${totalPages}...`);

            // PRE-PARSE HEADERS FOR THE PAGE
            const pageHeaders: { level: number, text: string, pos: number }[] = [];
            const lines = pageContent.split('\n');
            let currentPos = 0;
            for (const line of lines) {
                if (line.trim().startsWith('#')) {
                    const match = line.match(/^(#+)\s*(.*)$/);
                    if (match && match[1] && match[2]) {
                        pageHeaders.push({ 
                            level: match[1].length, 
                            text: match[2].trim(),
                            pos: currentPos
                        });
                    }
                }
                currentPos += line.length + 1;
            }

            const doc = new Document({ text: pageContent });
            const markdownNodes = await this.markdownParser.getNodesFromDocuments([doc]);

            const pageChunksNodes: any[] = [];
            const parentMap = new Map<any, string>();

            for (const node of markdownNodes) {
                if (strategy === 'HIERARCHICAL') {
                    const childNodes = await this.hierarchicalSplitter.getNodesFromDocuments([new Document({ text: node.text })]);
                    for (const child of childNodes) {
                        pageChunksNodes.push(child);
                        parentMap.set(child, node.text); 
                    }
                } else if (strategy === 'FULL_PAGE') {
                    pageChunksNodes.push(node);
                } else {
                    const sentenceNodes = await this.standardSplitter.getNodesFromDocuments([new Document({ text: node.text })]);
                    pageChunksNodes.push(...sentenceNodes);
                }
            }

            for (let i = 0; i < pageChunksNodes.length; i++) {
                const chunk = pageChunksNodes[i];
                const headers = this.extractHeadersContext(pageContent, chunk.text, pageHeaders);

                let overlapContent = "";
                if (i > 0) {
                    const prevChunk = pageChunksNodes[i - 1];
                    if (prevChunk) overlapContent += prevChunk.text.slice(-100);
                } else if (pageNum > 1 && allChunks.length > 0) {
                    const lastChunk = allChunks[allChunks.length - 1];
                    if (lastChunk) overlapContent += lastChunk.content.slice(-100);
                }

                if (i < pageChunksNodes.length - 1) {
                    const nextChunk = pageChunksNodes[i + 1];
                    if (nextChunk) overlapContent += nextChunk.text.slice(0, 100);
                }

                let content = chunk.text;
                if (strategy === 'CONTEXTUAL' && pageSummary) {
                    const contextPrefix = `[Context: ${headers.join(' > ')}${pageSummary ? ' | Page Summary: ' + pageSummary : ''}]\n`;
                    content = contextPrefix + content;
                }

                const chunkData: ChunkData = {
                    content: content,
                    title: path.parse(path.basename(sourceFile)).name,
                    sourceFile: sourceFile,
                    chunkIndex: allChunks.length,
                    chunkSize: content.length,
                    precedingHeaders: headers,
                    overlapContent: overlapContent,
                    documentHash: docHash,
                    processedAt: new Date().toISOString(),
                    pageNumber: pageNum,
                    pageChunkIndex: i,
                    totalPages: totalPages,
                    chunksInPage: pageChunksNodes.length,
                    chunkMethod: `page_aware_${strategy.toLowerCase()}`,
                    pageCharCount: pageContent.length,
                };
                
                if (parentMap.has(chunk)) {
                    chunkData.parentContent = parentMap.get(chunk);
                }
                if (pageSummary) {
                    chunkData.contextSummary = pageSummary;
                }

                allChunks.push(chunkData);
            }

            console.log(`    ✅ Page ${pageNum}: ${pageChunksNodes.length} chunks`);
        }

        console.log(`✅ Created ${allChunks.length} total chunks from ${pagesData.length} pages`);
        return allChunks;
    }
}
