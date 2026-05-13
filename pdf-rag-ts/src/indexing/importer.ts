export async function importToWeaviate(
    client: any,
    chunks: any[],
    collectionName: string
): Promise<void> {
    console.log(`Importing ${chunks.length} chunks to Weaviate...`);

    try {
        const batcher = client.batch.objectsBatcher();
        
        for (const chunk of chunks) {
            batcher.withObject({
                class: collectionName,
                properties: chunk,
            });
        }

        const results = await batcher.do();
        
        // Check for errors in results
        const errors = results.filter((r: any) => r.result?.errors);
        if (errors.length > 0) {
            console.warn(`⚠️ Warning: ${errors.length} chunks had errors during batch import.`);
            // In a real system, you might want to log the specific errors
        }

        console.log("✅ Import completed!");
    } catch (error) {
        console.error(`Error during import: ${error}`);
        console.log("Trying individual imports...");
        
        for (let i = 0; i < chunks.length; i++) {
            try {
                await client.data.creator()
                    .withClassName(collectionName)
                    .withProperties(chunks[i])
                    .do();
                if ((i + 1) % 10 === 0) {
                    console.log(`  Processed ${i + 1}/${chunks.length} chunks`);
                }
            } catch (chunkError) {
                console.error(`Error inserting chunk ${i}: ${chunkError}`);
            }
        }
    }
}
