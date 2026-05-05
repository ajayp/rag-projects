from typing import List, Dict, Any


def import_to_weaviate(client, chunks: List[Dict[str, Any]], collection_name: str) -> None:
    print(f"Importing {len(chunks)} chunks to Weaviate...")

    collection = client.collections.get(collection_name)

    try:
        with collection.batch.dynamic() as batch:
            for i, chunk in enumerate(chunks):
                batch.add_object(properties=chunk)
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(chunks)} chunks")
        print("✅ Import completed!")
    except Exception as e:
        print(f"Error during import: {e}")
        print("Trying individual imports...")
        for i, chunk in enumerate(chunks):
            try:
                collection.data.insert(chunk)
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(chunks)} chunks")
            except Exception as chunk_error:
                print(f"Error inserting chunk {i}: {chunk_error}")
