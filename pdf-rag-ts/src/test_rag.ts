import { LocalRAGSystem } from './index.js';
import * as dotenv from 'dotenv';

dotenv.config();

async function main() {
    console.log("Initializing Local RAG System (TS)...");
    try {
        const options: any = {};
        if (process.env.LLAMAPARSE_API_KEY) {
            options.llamaParseApiKey = process.env.LLAMAPARSE_API_KEY;
        }
        const rag = new LocalRAGSystem(options);
        
        console.log("RAG System initialized successfully!");
        
        // Note: This won't actually do much without Ollama and Weaviate running,
        // but it verifies the classes can be instantiated and basic setup logic runs.
        
        // const stats = await rag.getDocumentStats();
        // console.log("Current stats:", stats);
        
    } catch (error) {
        console.error("Failed to initialize RAG system:", error);
    }
}

main();
