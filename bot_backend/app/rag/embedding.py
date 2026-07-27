import litellm
import asyncio
import os
# Local embedding fallback using sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # Will be installed when needed

# Drop unsupported params (e.g., dimensions) for OpenAI embedding models
litellm.drop_params = True

class EmbeddingModel:
    """
    Turns text into vectors (lists of numbers) so we can compare how
    similar two pieces of text are by comparing their vectors.

    Uses Google's Gemini embedding API (hosted, no local model to download).
    """

    def __init__(self, model_name: str = "gemini/text-embedding-004", dimensions: int = 768):
        self.model_name = model_name
        self.dimensions = dimensions
        # Initialize local model lazily
        self._local_model = None

    async def embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Embeds many chunks at once — used during ingestion."""
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            try:
                # Try OpenAI via litellm first
                response = await litellm.aembedding(
                    model=self.model_name,
                    input=batch,
                    # dimensions param not supported for OpenAI; litellm will drop it via litellm.drop_params
                    task_type="RETRIEVAL_DOCUMENT",
                )
                embeddings.extend(item["embedding"] for item in response.data)
            except Exception as e:
                # If OpenAI fails (e.g., quota), fall back to local model
                if SentenceTransformer is None:
                    raise RuntimeError("sentence-transformers not installed; cannot fallback for embeddings")
                if self._local_model is None:
                    # Load a model that outputs 768 dimensions to match the DB index
                    self._local_model = SentenceTransformer("all-mpnet-base-v2")
                # Compute local embeddings synchronously
                local_embs = self._local_model.encode(batch, normalize_embeddings=True)
                # Convert NumPy float32 vectors to native Python floats for BSON storage
                for vec in local_embs:
                    embeddings.append([float(v) for v in vec])
                import logging
                logging.getLogger("uvicorn").info("Used local SentenceTransformer embeddings as fallback")
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embeds a single piece of text — used for a user's question."""
        try:
            response = await litellm.aembedding(
                model=self.model_name,
                input=[text],
                task_type="RETRIEVAL_QUERY",
            )
            return response.data[0]["embedding"]
        except Exception as e:
            # Fallback to local model for single query
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed; cannot fallback for query embedding")
            if self._local_model is None:
                self._local_model = SentenceTransformer("all-mpnet-base-v2")
            local_vec = self._local_model.encode([text], normalize_embeddings=True)[0]
            # Convert to native Python floats
            python_vec = [float(v) for v in local_vec]
            import logging
            logging.getLogger("uvicorn").info("Used local SentenceTransformer for query embedding fallback")
            return python_vec

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