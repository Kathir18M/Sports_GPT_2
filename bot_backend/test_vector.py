import sys
import os
import asyncio

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from app.rag.loader import PDFLoader
from app.rag.splitter import DocumentSplitter
from app.rag.embedding import EmbeddingModel
from app.rag.vectorstore import VectorDatabase
from app.db.mongo_db import connect_to_mongo, close_mongo_connection


async def main():
    print("=" * 60)
    print("Connecting to MongoDB...")
    print("=" * 60)
    await connect_to_mongo()

    print("=" * 60)
    print("Loading Documents...")
    print("=" * 60)

    loader = PDFLoader()
    documents = loader.load_documents()

    print(f"Documents : {len(documents)}")


    print("\n" + "=" * 60)
    print("Chunking Documents...")
    print("=" * 60)

    splitter = DocumentSplitter()
    chunks = splitter.split_documents(documents)

    print(f"Chunks : {len(chunks)}")


    print("\n" + "=" * 60)
    print("Creating Embeddings...")
    print("=" * 60)

    embedding_model = EmbeddingModel()

    texts = embedding_model.get_texts(chunks)

    embeddings = embedding_model.create_embeddings(texts)

    embedded_documents = embedding_model.build_documents(
        chunks,
        embeddings
    )

    print(f"Embeddings : {len(embedded_documents)}")


    print("\n" + "=" * 60)
    print("Creating Vector Database...")
    print("=" * 60)

    db = VectorDatabase()

    db.create_index()

    # Clear the collection first for clean testing
    from app.db.mongo_db import get_vector_collection
    collection = get_vector_collection()
    await collection.delete_many({})
    print("Cleared existing test vector documents from MongoDB.")

    await db.add_documents(embedded_documents)


    print("\n" + "=" * 60)
    print("Testing Search...")
    print("=" * 60)

    query = "Who is the No.1 ODI Team?"

    query_embedding = embedding_model.model.encode(
        query,
        convert_to_numpy=True
    )

    results = await db.search(
        query_embedding,
        top_k=5
    )

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")

        print("-" * 40)

        print("Source :", result["source"])

        print("Score :", result["score"])

        print("Text :")

        print(result["text"][:300])


    print("\n" + "=" * 60)
    print("Saving Vector Database...")
    print("=" * 60)

    await db.save()

    print("\n" + "=" * 60)
    print("Loading Vector Database...")
    print("=" * 60)

    await db.load()

    print("\nClosing MongoDB Connection...")
    await close_mongo_connection()

    print("\n ALL TESTS PASSED SUCCESSFULLY ")


if __name__ == "__main__":
    asyncio.run(main())
