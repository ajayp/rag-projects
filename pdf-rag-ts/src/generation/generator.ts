export class AnswerGenerator {
    private ollamaUrl: string;
    private generativeModel: string;

    constructor(ollamaUrl: string, generativeModel: string) {
        this.ollamaUrl = ollamaUrl;
        this.generativeModel = generativeModel;
    }

    async generate(prompt: string): Promise<string> {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000);

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
                throw new Error(`Ollama error: ${response.statusText}`);
            }

            const data = await response.json() as { response?: string };
            return (data.response || "").trim();
        } catch (error) {
            clearTimeout(timeoutId);
            throw error;
        }
    }
}
