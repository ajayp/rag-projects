export class QueryAugmenter {
    private ollamaUrl: string;
    private generativeModel: string;
    private rewriteModel: string;

    constructor(ollamaUrl: string, generativeModel: string, rewriteModel: string) {
        this.ollamaUrl = ollamaUrl;
        this.generativeModel = generativeModel;
        this.rewriteModel = rewriteModel;
    }

    async rewriteQuery(question: string): Promise<string> {
        const prompt = `You are a search query optimizer. Your task is to provide 3-5 high-quality keywords or technical synonyms that would help find the answer to the user's question in a technical document.

Rules:
- Output ONLY the keywords, separated by spaces.
- Do NOT repeat words already in the question.
- Do NOT include conversational filler.
- Focus on technical terminology.

Question: what is RAG
Keywords: retrieval augmented generation vector database embeddings

Question: how does chunking work
Keywords: splitting segmentation tokenization overlap context

Question: what are the requirements
Keywords: prerequisites specifications criteria qualifications

Question: ${question}
Keywords:`;

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        try {
            const response = await fetch(`${this.ollamaUrl}/api/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: this.rewriteModel,
                    prompt: prompt,
                    stream: false,
                }),
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                return question;
            }

            const data = await response.json() as { response?: string };
            const expansion = (data.response || "").trim().toLowerCase();
            
            // Filter out any common conversational words the model might have included
            const filteredExpansion = expansion.replace(/^(keywords:|expanded:)/i, '').trim();
            
            const rewritten = filteredExpansion ? `${question} ${filteredExpansion}` : question;
            console.log(`🔄 Query expanded: '${question}' → '${rewritten}'`);
            return rewritten;
        } catch (error) {
            clearTimeout(timeoutId);
            return question;
        }
    }

    async generateHypotheticalAnswer(question: string): Promise<string> {
        const prompt = `Write a short, technically detailed passage (2-4 sentences) that directly answers the following question. Write as if you are the document being searched — use specific terms, model names, and technical details that would appear in a technical document.

Question: ${question}
Passage:`;

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        try {
            const response = await fetch(`${this.ollamaUrl}/api/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: this.generativeModel,
                    prompt: prompt,
                    stream: false,
                }),
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                return question;
            }

            const data = await response.json() as { response?: string };
            const hypothetical = (data.response || question).trim();
            console.log(`💭 HyDE passage: ${hypothetical.substring(0, 100)}...`);
            return hypothetical;
        } catch (error) {
            clearTimeout(timeoutId);
            return question;
        }
    }
}
