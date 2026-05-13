import { objToDict, type Chunk } from '../utils/helpers.js';

export class Searcher {
    private client: any;
    private collectionName: string;
    private minContentLength: number;

    constructor(client: any, collectionName: string, minContentLength: number = 150) {
        this.client = client;
        this.collectionName = collectionName;
        this.minContentLength = minContentLength;
    }

    async search(
        query: string,
        limit: number = 5,
        sourceFile?: string,
        alpha: number = 0.75,
        vector?: number[]
    ): Promise<Chunk[]> {
        let builder = this.client.graphql
            .get()
            .withClassName(this.collectionName)
            .withFields('content title sourceFile chunkIndex precedingHeaders pageNumber pageChunkIndex totalPages _additional { score distance }')
            .withHybrid({
                query: query,
                alpha: alpha,
                ...(vector ? { vector: vector } : {})
            })
            .withLimit(limit * 3);

        if (sourceFile) {
            builder = builder.withWhere({
                path: ["sourceFile"],
                operator: "Equal",
                valueString: sourceFile
            });
        }

        const response = await builder.do();
        const objects = response.data.Get[this.collectionName] || [];
        
        const results: Chunk[] = [];
        for (const obj of objects) {
            const chunk = objToDict({ properties: obj, metadata: obj._additional });
            if (chunk.content.length >= this.minContentLength) {
                results.push(chunk);
                if (results.length >= limit) break;
            }
        }

        if (results.length === 0) {
            return objects.slice(0, limit).map((obj: any) => 
                objToDict({ properties: obj, metadata: obj._additional })
            );
        }

        return results;
    }

    async sectionFilteredSearch(
        query: string,
        requiredSections: string[],
        limit: number = 5,
        sourceFile?: string,
        alpha: number = 0.75,
        vector?: number[]
    ): Promise<Chunk[]> {
        let builder = this.client.graphql
            .get()
            .withClassName(this.collectionName)
            .withFields('content title sourceFile chunkIndex precedingHeaders pageNumber pageChunkIndex totalPages _additional { score distance }')
            .withHybrid({
                query: query,
                alpha: alpha,
                ...(vector ? { vector: vector } : {})
            })
            .withLimit(limit * 3);

        if (sourceFile) {
            builder = builder.withWhere({
                path: ["sourceFile"],
                operator: "Equal",
                valueString: sourceFile
            });
        }

        const response = await builder.do();
        const objects = response.data.Get[this.collectionName] || [];
        
        const results: Chunk[] = [];
        for (const obj of objects) {
            const headers: string[] = obj.precedingHeaders || [];
            if (requiredSections.some(s => headers.some(h => h.toLowerCase().includes(s.toLowerCase())))) {
                results.push(objToDict({ properties: obj, metadata: obj._additional }));
                if (results.length >= limit) break;
            }
        }

        return results;
    }

    async searchByPage(
        query: string,
        pageNumbers?: number[],
        limit: number = 5,
        sourceFile?: string,
        alpha: number = 0.75,
        vector?: number[]
    ): Promise<Chunk[]> {
        let builder = this.client.graphql
            .get()
            .withClassName(this.collectionName)
            .withFields('content title sourceFile chunkIndex precedingHeaders pageNumber pageChunkIndex totalPages _additional { score distance }')
            .withHybrid({
                query: query,
                alpha: alpha,
                ...(vector ? { vector: vector } : {})
            })
            .withLimit(limit * 3);

        if (sourceFile) {
            builder = builder.withWhere({
                path: ["sourceFile"],
                operator: "Equal",
                valueString: sourceFile
            });
        }

        const response = await builder.do();
        const objects = response.data.Get[this.collectionName] || [];
        
        if (!pageNumbers || pageNumbers.length === 0) {
            return objects.slice(0, limit).map((obj: any) => 
                objToDict({ properties: obj, metadata: obj._additional })
            );
        }

        const results: Chunk[] = [];
        for (const obj of objects) {
            if (pageNumbers.includes(obj.pageNumber)) {
                results.push(objToDict({ properties: obj, metadata: obj._additional }));
                if (results.length >= limit) break;
            }
        }

        return results;
    }
}
