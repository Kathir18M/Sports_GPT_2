import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
import litellm

async def test():
    try:
        response = await litellm.aembedding(
            model="gemini/text-embedding-004",
            input=["hi"],
            task_type="RETRIEVAL_QUERY",
        )
        print("Success 004:", len(response.data[0]["embedding"]))
    except Exception as e:
        print("Error 004:", repr(e))

    try:
        response = await litellm.aembedding(
            model="gemini/embedding-001",
            input=["hi"],
            task_type="RETRIEVAL_QUERY",
        )
        print("Success 001:", len(response.data[0]["embedding"]))
    except Exception as e:
        print("Error 001:", repr(e))


asyncio.run(test())
