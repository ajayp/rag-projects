export async function setupOllamaModels(
    ollamaUrl: string,
    embeddingModel: string,
    generativeModel: string,
    rewriteModel: string
): Promise<void> {
    console.log("Setting up Ollama models...");
    try {
        const tagsResponse = await fetch(`${ollamaUrl}/api/tags`);
        if (tagsResponse.ok) {
            const data = await tagsResponse.json() as { models: { name: string }[] };
            const models: string[] = data.models.map((m: any) => m.name);

            if (!models.some(m => m.includes(embeddingModel))) {
                console.log(`Pulling embedding model: ${embeddingModel}`);
                await pullOllamaModel(ollamaUrl, embeddingModel);
            }

            if (!models.some(m => m.includes(generativeModel))) {
                console.log(`Pulling generative model: ${generativeModel}`);
                await pullOllamaModel(ollamaUrl, generativeModel);
            }

            if (!models.some(m => m.includes(rewriteModel))) {
                console.log(`Pulling rewrite model: ${rewriteModel}`);
                await pullOllamaModel(ollamaUrl, rewriteModel);
            }

            console.log("✅ Ollama models ready!");
        } else {
            console.warn("⚠️ Could not connect to Ollama. Make sure it's running on port 11434");
        }
    } catch (error) {
        console.error(`⚠️ Ollama setup error: ${error}`);
    }
}

async function pullOllamaModel(ollamaUrl: string, modelName: string): Promise<void> {
    try {
        const response = await fetch(`${ollamaUrl}/api/pull`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name: modelName }),
        });

        if (!response.ok) {
            throw new Error(`Failed to pull model: ${response.statusText}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
            throw new Error("Response body is not readable");
        }

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.trim() === '') continue;
                try {
                    const data = JSON.parse(line);
                    if (data.status) {
                        console.log(`  ${data.status}`);
                    }
                    if (data.status === 'success') {
                        return;
                    }
                } catch (e) {
                    // Ignore partial JSON
                }
            }
        }
    } catch (error) {
        console.error(`Error pulling model ${modelName}: ${error}`);
    }
}
