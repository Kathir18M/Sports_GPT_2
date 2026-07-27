from app.rag.embedding import EmbeddingModel
from app.rag.vectorstore import VectorDatabase
import logging

class Retriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_db: VectorDatabase):
        self.embedding_model = embedding_model
        self.vector_db = vector_db

    # ...

    async def retrieve(self, query: str, top_k: int = 5):
        # Get the query embedding using the correct method with error handling
        try:
            query_embedding = await self.embedding_model.embed_query(query)
        except Exception as e:
            logging.getLogger("uvicorn").warning(f"Failed to embed query: {e}. Returning empty results.")
            return []
        # Perform the similarity search and return results (ensure a list)
        results = await self.vector_db.search(query_embedding, top_k)
        return results if results else []

    def build_context(self, results):
        """Construct a plain‑text context from retrieved chunks.
        The list `results` contains dicts with at least ``source`` and ``text`` keys.
        Returns a single string that concatenates each chunk with a header.
        """
        context = ""
        for result in results:
            source = result.get('source', 'unknown')
            text = result.get('text', '')
            context += f"Source: {source}\n{text}\n\n"
        return context



