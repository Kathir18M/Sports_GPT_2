import litellm
import asyncio
import os

# Drop unsupported params (e.g., dimensions) for OpenAI embedding models
litellm.drop_params = True

class EmbeddingModel:
    """
    Turns text into vectors (lists of numbers) so we can compare how
    similar two pieces of text are by comparing their vectors.

    Uses Google's Gemini embedding API (hosted, no local model to download).
    """

    def __init__(self, model_name: str = "gemini/gemini-embedding-001", dimensions: int = 768):
        self.model_name = model_name
        self.dimensions = dimensions

    async def embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Embeds many chunks at once — used during ingestion."""
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = await litellm.aembedding(
                model=self.model_name,
                input=batch,
                # dimensions param not supported for OpenAI; litellm will drop it via litellm.drop_params
                task_type="RETRIEVAL_DOCUMENT",
            )
            # Truncate to self.dimensions (768) to match the Atlas vector index
            embeddings.extend(item["embedding"][:self.dimensions] for item in response.data)
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embeds a single piece of text — used for a user's question."""
        response = await litellm.aembedding(
            model=self.model_name,
            input=[text],
            task_type="RETRIEVAL_QUERY",
        )
        # Truncate to self.dimensions (768) to match the Atlas vector index
        return response.data[0]["embedding"][:self.dimensions]

    # Compatibility helpers for existing synchronous code
    def get_texts(self, chunks):
        """Extract raw text from chunk dicts (sync API)."""
        return [chunk.get("text", "") for chunk in chunks]

    def create_embeddings(self, texts: list[str]):
        """Sync wrapper that runs async embed_texts and returns embeddings."""
        if not texts:
            return []
        return asyncio.run(self.embed_texts(texts))

    def build_documents(self, chunks, embeddings):
        """Combine chunks with embeddings into document dicts for storage."""
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            documents.append({
                "source": chunk.get("source", "unknown"),
                "text": chunk.get("text", ""),
                "embedding": embedding,
            })
        return documents