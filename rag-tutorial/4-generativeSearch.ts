import { withWeaviateClient, promptUser, COLLECTION_NAME, DEFAULT_BOOK_QUERY } from './utils/utils.ts';

withWeaviateClient(async (client) => {
    const booksCollection = client.collections.get(COLLECTION_NAME);

    const userQuery = (await promptUser('Describe the book you are looking for (e.g., genre, plot, themes), or hit return for the default query: ')) || DEFAULT_BOOK_QUERY;

    console.log(`Searching for books related to: ${userQuery}, please wait...`);
    const responses = await booksCollection.generate.nearText(
        userQuery,
        { singlePrompt: `Explain why this book might appeal to them. The book's title is {title}, with a description: {description}, and is in the genre: {categories}.` },
        { limit: 4 }
    );

    console.log('\nRecommendations based on your interest:\n');
    for (const response of responses.objects) {
        console.log(response.generative?.text ?? '(no generative response)');
        console.log('\n');
    }
}).catch(console.error);
