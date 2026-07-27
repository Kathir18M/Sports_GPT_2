# app/routes/ingest.py
"""FastAPI router that forces a full PDF ingestion.
Used when the automatic startup ingestion is skipped or when you need to re‑ingest after changing data.
"""

from fastapi import APIRouter, HTTPException

from app.rag.loader import PDFLoader
from app.rag.splitter import DocumentSplitter
from app.rag.embedding import EmbeddingModel
from app.rag.vectorstore import VectorDatabase

router = APIRouter(prefix="/ingest", tags=["Ingest"])

@router.post("/", summary="Force a full PDF ingestion")
async def ingest_now():
    """Load PDFs, split, embed and store them in MongoDB.
    Returns a message with the number of inserted documents.
    """
    try:
        # Load PDFs from the configured folder
        loader = PDFLoader()
        documents = loader.load_documents()
        if not documents:
            raise HTTPException(status_code=400, detail="No PDF files found in the data folder.")

        # Chunk the documents
        splitter = DocumentSplitter()
        chunks = splitter.split_documents(documents)

        # Create embeddings
        embedder = EmbeddingModel()
        texts = embedder.get_texts(chunks)
        # Use the async embed_texts coroutine directly in async context
        embeddings = await embedder.embed_texts(texts)

        # Build final document dicts (including metadata)
        embedded_documents = embedder.build_documents(chunks, embeddings)

        # Store them in MongoDB Atlas / local DB
        vectordb = VectorDatabase()
        await vectordb.add_documents(embedded_documents)

        return {"msg": f"Ingested {len(embedded_documents)} vector documents successfully."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
