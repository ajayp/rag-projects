import * as http from 'http';
import { LocalRAGSystem } from './index.js';
import type { ChunkingStrategy } from './chunking/chunker.js';

const PORT = 3000;
const rag = new LocalRAGSystem();

const server = http.createServer(async (req, res) => {
    // CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
    }

    const url = new URL(req.url || '', `http://localhost:${PORT}`);

    try {
        if (url.pathname === '/process' && req.method === 'POST') {
            const body = await getBody(req);
            const { filePath, strategy, useLlamaParse } = JSON.parse(body);
            try {
                await rag.processDocument(filePath, { strategy, useLlamaParse });
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ status: 'success' }));
            } catch (err: any) {
                if (err.message?.startsWith('ALREADY_EXISTS:')) {
                    const name = err.message.split(':')[1];
                    res.writeHead(409, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: `'${name}' is already indexed. Reset the collection first.` }));
                } else {
                    throw err;
                }
            }
        } 
        else if (url.pathname === '/query' && req.method === 'POST') {
            const body = await getBody(req);
            const { question, options } = JSON.parse(body);
            const answer = await rag.askQuestion(question, options);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ answer }));
        }
        else if (url.pathname === '/stats' && req.method === 'GET') {
            const stats = await rag.getDocumentStats();
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(stats));
        }
        else if (url.pathname === '/reset' && req.method === 'POST') {
            await rag.reset();
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ status: 'success' }));
        }
        else {
            res.writeHead(404);
            res.end();
        }
    } catch (error: any) {
        console.error(error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message }));
    }
});

function getBody(req: http.IncomingMessage): Promise<string> {
    return new Promise((resolve, reject) => {
        let body = '';
        req.on('data', chunk => body += chunk.toString());
        req.on('end', () => resolve(body));
        req.on('error', reject);
    });
}

server.listen(PORT, () => {
    console.log(`🚀 TS RAG Server running at http://localhost:${PORT}`);
});

const shutdown = () => {
    server.close(() => process.exit(0));
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
