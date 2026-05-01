import weaviate from 'weaviate-client';
import readline from 'node:readline';
import { once } from 'node:events';

export const COLLECTION_NAME = 'Books';
export const DEFAULT_BOOK_QUERY = 'I want a space based story about heroes, villains, rebels, and tyrants, with pedal-bin droids.';

function safeParse(value: string, parseFn: (s: string) => number): number | null {
    if (!value) return null;
    const num = parseFn(value);
    return isNaN(num) ? null : num;
}

export const safeParseInt   = (value: string) => safeParse(value, s => parseInt(s, 10));
export const safeParseFloat = (value: string) => safeParse(value, parseFloat);

export async function promptUser(message: string): Promise<string> {
    process.stdout.write(message);
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    try {
        const [input] = await once(rl, 'line');
        return (input as string).trim();
    } finally {
        rl.close();
    }
}

type WeaviateClient = Awaited<ReturnType<typeof weaviate.connectToLocal>>;

export async function withWeaviateClient(fn: (client: WeaviateClient) => Promise<void>): Promise<void> {
    let client: WeaviateClient | undefined;
    try {
        client = await weaviate.connectToLocal();
        console.log('Successfully connected to Weaviate.');
        await fn(client);
    } catch (error) {
        console.error('An error occurred:', error);
    } finally {
        if (client) {
            await client.close();
            console.log('Weaviate client closed.');
        }
    }
}
