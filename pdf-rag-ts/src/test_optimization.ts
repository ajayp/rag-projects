import { LocalRAGSystem } from './index.js';
import * as dotenv from 'dotenv';
import * as path from 'path';

dotenv.config();

async function main() {
    console.log("🚀 Testing Optimized Contextual Chunking...");
    try {
        const rag = new LocalRAGSystem();
        
        // We'll use a local markdown parsing strategy to avoid hitting LlamaParse for this test
        // but still use CONTEXTUAL strategy which triggers the summary parallelization
        const filePath = path.join(process.cwd(), '../README.md'); 
        
        console.log(`\nTesting with file: ${filePath}`);
        const startTime = Date.now();
        
        await rag.processDocument(filePath, { 
            useLlamaParse: false, 
            strategy: 'CONTEXTUAL' 
        });
        
        const duration = (Date.now() - startTime) / 1000;
        console.log(`\n✨ Test complete in ${duration.toFixed(2)} seconds!`);
        
    } catch (error) {
        console.error("❌ Test failed:", error);
    }
}

main();
