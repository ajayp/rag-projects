import { withWeaviateClient, promptUser, COLLECTION_NAME, DEFAULT_BOOK_QUERY } from './utils/utils.ts';

withWeaviateClient(async (client) => {
    const booksCollection = client.collections.get(COLLECTION_NAME);

    const userQuery = (await promptUser("Describe the book you're looking for (e.g., genre, plot, themes), or hit return for the default query: ")) || DEFAULT_BOOK_QUERY;

    const responses = await booksCollection.query.nearText(userQuery, {
        limit: 3,
        returnProperties: ['title', 'authors', 'description', 'average_rating', 'categories'],
    });

    console.log('\nRecommendations:\n');
    for (const response of responses.objects) {
        console.log(`Title: ${response.properties.title ?? 'N/A'}`);
        console.log(`Description: ${response.properties.description ?? 'N/A'}`);
        console.log(`Average Rating: ${response.properties.average_rating ?? 'N/A'}`);
        console.log(`Category: ${response.properties.categories ?? 'N/A'}`);
        console.log('\n');
    }
}).catch(console.error);
