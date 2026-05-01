import * as fs from 'fs';
import * as csv from 'csv-parse';
import weaviate from 'weaviate-client';
import { safeParseInt, safeParseFloat, COLLECTION_NAME } from './utils/utils.ts';

interface BookProperties {
    isbn13: string;
    isbn10: string;
    title: string;
    subtitle: string;
    authors: string;
    categories: string;
    thumbnail: string;
    description: string;
    published_year: number | null;
    average_rating: number | null;
    num_pages: number | null;
    ratings_count: number | null;
}

const CSV_COLUMNS = [
    'isbn13', 'isbn10', 'title', 'subtitle', 'authors', 'categories',
    'thumbnail', 'description', 'published_year', 'average_rating',
    'num_pages', 'ratings_count',
];

async function populateBooks(): Promise<void> {
    const client = await weaviate.connectToLocal();
    const booksCollection = client.collections.get(COLLECTION_NAME);

    const fileStream = fs.createReadStream('./dataset/7k-books-kaggle.csv');
    const parser = csv.parse({ from_line: 1, columns: CSV_COLUMNS });

    const BATCH_SIZE = 500;
    const batchRecords: BookProperties[] = [];

    async function insertBatch(records: BookProperties[]): Promise<void> {
        if (records.length === 0) return;
        try {
            await booksCollection.data.insertMany(records as Record<string, any>[]);
            console.log(`Successfully inserted batch of ${records.length} records.`);
        } catch (error) {
            console.error('Errors inserting batch:', error);
            throw error;
        }
    }

    try {
        fileStream.pipe(parser);

        for await (const book of parser) {
            const bookProperties: BookProperties = {
                isbn13: book.isbn13,
                isbn10: book.isbn10,
                title: book.title,
                subtitle: book.subtitle,
                authors: book.authors,
                categories: book.categories,
                thumbnail: book.thumbnail,
                description: book.description,
                published_year: safeParseInt(book.published_year),
                average_rating: safeParseFloat(book.average_rating),
                num_pages: safeParseInt(book.num_pages),
                ratings_count: safeParseInt(book.ratings_count),
            };

            batchRecords.push(bookProperties);
            if (batchRecords.length >= BATCH_SIZE) {
                await insertBatch(batchRecords);
                batchRecords.length = 0;
            }
        }

        await insertBatch(batchRecords);
    } catch (error) {
        console.error('Error processing CSV:', error);
    } finally {
        console.log('Finished populating books.');
        fileStream.destroy();
        await client.close();
    }
}

populateBooks().catch(console.error);
