import os

from app.rag.loader import PDFLoader
from app.rag.splitter import DocumentSplitter
from app.rag.embedding import EmbeddingModel
from app.rag.vectorstore import VectorDatabase
from app.rag.retriever import Retriever


INDEX_PATH = "vectorstore/vector.index"
METADATA_PATH = "vectorstore/metadata.pkl"


class RAGPipeline:

    def __init__(self):

        self.embedding_model = None
        self.vector_db = None
        self.retriever = None
        self.is_ready = False
        self.is_initializing = False

    async def initialize(self):
        if self.is_ready or self.is_initializing:
            return

        self.is_initializing = True

        print("=" * 60)
        print("Initializing RAG Pipeline in background...")
        print("=" * 60)

        self.embedding_model = EmbeddingModel()

        self.vector_db = VectorDatabase()

        from app.db.mongo_db import get_vector_collection
        try:
            collection = get_vector_collection()
            count = await collection.count_documents({})
        except Exception as e:
            print(f"Error checking MongoDB vector collection: {e}")
            count = 0

        if count > 0:
            print("Found existing documents in MongoDB. Skipping PDF ingestion.")
            await self.vector_db.load()
        else:
            print("MongoDB vector collection is empty. Starting initial PDF loading and ingestion...")

            loader = PDFLoader()
            documents = loader.load_documents()

            self.vector_db.create_index()

            if documents:
                splitter = DocumentSplitter()
                chunks = splitter.split_documents(documents)

                texts = self.embedding_model.get_texts(chunks)

                embeddings = await self.embedding_model.embed_texts(texts)

                embedded_documents = self.embedding_model.build_documents(
                    chunks,
                    embeddings
                )

                await self.vector_db.add_documents(
                    embedded_documents
                )
            else:
                print("Warning: No documents loaded. Empty Vector Database created.")

            await self.vector_db.save()

       
        self.retriever = Retriever(
            self.embedding_model,
            self.vector_db
        )

        self.is_ready = True
        self.is_initializing = False

        print("=" * 60)
        print("SportsBot RAG Pipeline Ready")
        print("=" * 60)


rag_pipeline = RAGPipeline()