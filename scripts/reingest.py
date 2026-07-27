import os, sys, asyncio
# Ensure PDF_DATA_PATH points to the correct data folder
os.environ['PDF_DATA_PATH'] = r'C:/Users/kathi/OneDrive/Attachments/Documents/sports_gpt/data'
# Add the bot_backend package to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bot_backend')))
# Import MongoDB connection helpers
from app.db.mongo_db import connect_to_mongo, close_mongo_connection
from app.rag.loader import PDFLoader
from app.rag.splitter import DocumentSplitter
from app.rag.embedding import EmbeddingModel
from app.rag.vectorstore import VectorDatabase

async def main():
    # Connect to MongoDB before ingesting
    await connect_to_mongo()
    loader = PDFLoader(data_folder=os.getenv('PDF_DATA_PATH'))
    docs = loader.load_documents()
    if not docs:
        print('⚠️ No PDFs found – check the ./data folder')
        await close_mongo_connection()
        return
        # duplicate return removed

    splitter = DocumentSplitter()
    chunks = splitter.split_documents(docs)

    embedder = EmbeddingModel()
    texts = embedder.get_texts(chunks)
    embeddings = await embedder.embed_texts(texts)

    vectordb = VectorDatabase()
    await vectordb.add_documents(embedder.build_documents(chunks, embeddings))

    print(f'✅ Ingested {len(chunks)} chunks ({len(embeddings)} embeddings)')
    # Close DB connection after ingestion
    await close_mongo_connection()

if __name__ == '__main__':
    asyncio.run(main())
