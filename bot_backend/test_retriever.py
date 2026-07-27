import asyncio
from app.rag.loader import PDFLoader
from app.rag.splitter import DocumentSplitter
from app.rag.embedding import EmbeddingModel
from app.rag.vectorstore import VectorDatabase
from app.rag.retriever import Retriever
from app.db.mongo_db import connect_to_mongo, close_mongo_connection


async def main():
    print("Connecting to MongoDB...")
    await connect_to_mongo()

    loader = PDFLoader()
    documents = loader.load_documents()

    splitter = DocumentSplitter()
    chunks = splitter.split_documents(documents)

    embedding_model = EmbeddingModel()

    texts = embedding_model.get_texts(chunks)

    embeddings = embedding_model.create_embeddings(texts)

    embedded_documents = embedding_model.build_documents(
        chunks,
        embeddings
    )

    vector_db = VectorDatabase()

    vector_db.create_index()

    # Clear first for clean testing
    from app.db.mongo_db import get_vector_collection
    collection = get_vector_collection()
    await collection.delete_many({})
    print("Cleared existing test vector documents from MongoDB.")

    await vector_db.add_documents(embedded_documents)

    retriever = Retriever(
        embedding_model,
        vector_db
    )

    query = "Who is the No.1 ODI Team?"

    results = await retriever.retrieve(query)

    context = retriever.build_context(results)

    print("=" * 70)
    print("Retrieved Context")
    print("=" * 70)

    print(context)

    print("Closing MongoDB Connection...")
    await close_mongo_connection()


if __name__ == "__main__":
    asyncio.run(main())