import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from app.rag.embedding import EmbeddingModel

async def test():
    e = EmbeddingModel()
    emb = await e.embed_query('hi')
    print(len(emb))

asyncio.run(test())
