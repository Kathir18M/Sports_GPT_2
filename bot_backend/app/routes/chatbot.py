import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.ai.llm import (
    generate_response,
    stream_response
)

from app.rag.startup import rag_pipeline
from app.prompts.rag_prompt import build_prompt

router = APIRouter()


async def ensure_rag_ready():
    if not rag_pipeline.is_ready:
        if not rag_pipeline.is_initializing:
            asyncio.create_task(rag_pipeline.initialize())
        while not rag_pipeline.is_ready:
            await asyncio.sleep(0.1)


@router.get("/chat")
async def chat(q: str, mode: str):

    await ensure_rag_ready()
    results = await rag_pipeline.retriever.retrieve(q)

    context = rag_pipeline.retriever.build_context(results)

    prompt = build_prompt(context, q)

    return generate_response(prompt, mode)


@router.get("/chat/stream")
async def chat_stream(q: str, mode: str):

    await ensure_rag_ready()
    results = await rag_pipeline.retriever.retrieve(q)


    context = rag_pipeline.retriever.build_context(results)

    
    prompt = build_prompt(context, q)


    return StreamingResponse(
        stream_response(prompt, mode),
        media_type="text/plain"
    )