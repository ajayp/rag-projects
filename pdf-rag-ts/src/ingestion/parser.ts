import * as fs from 'fs';
import { LlamaParseReader } from 'llamaindex';

export interface PageData {
    content: string;
    pageNumber: number;
    sourceFile: string;
    totalPages: number;
}

export class DocumentParser {
    private llamaParseApiKey: string | undefined;

    constructor(llamaParseApiKey?: string) {
        this.llamaParseApiKey = llamaParseApiKey || process.env.LLAMAPARSE_API_KEY;
    }

    async parseWithLlamaParse(filePath: string): Promise<PageData[]> {
        if (!this.llamaParseApiKey) {
            throw new Error("LlamaParse API key required for document parsing");
        }

        const reader = new LlamaParseReader({
            apiKey: this.llamaParseApiKey,
            resultType: "markdown",
            splitByPage: true,
            verbose: true,
        });

        const documents = await reader.loadData(filePath);
        if (!documents || documents.length === 0) {
            throw new Error("Failed to parse document or received empty response.");
        }

        console.log(`📄 LlamaParse returned ${documents.length} pages`);

        const pagesData: PageData[] = [];
        for (let i = 0; i < documents.length; i++) {
            const doc = documents[i];
            if (doc) {
                const pageContent = doc.text.trim();
                if (pageContent) {
                    pagesData.push({
                        content: pageContent,
                        pageNumber: i + 1,
                        sourceFile: filePath,
                        totalPages: documents.length,
                    });
                    console.log(`  📄 Page ${i + 1}: ${pageContent.length} characters`);
                }
            }
        }

        console.log(`✅ Processed ${pagesData.length} non-empty pages`);
        return pagesData;
    }

    async parseLocalMarkdown(filePath: string): Promise<PageData[]> {
        const content = fs.readFileSync(filePath, 'utf-8');
        return [{
            content: content,
            pageNumber: 1,
            sourceFile: filePath,
            totalPages: 1,
        }];
    }
}
