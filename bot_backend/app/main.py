import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.chatbot import router as chatbot_router

from app.rag.startup import rag_pipeline
from app.db.mongo_db import connect_to_mongo, close_mongo_connection, db_instance, get_vector_collection

os.makedirs("outputs", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("Starting SportsBot...")
    print("=" * 60)

    try:
        await connect_to_mongo()
    except Exception as e:
        print(f"Failed to connect to MongoDB on startup: {e}")

    asyncio.create_task(rag_pipeline.initialize())

    yield

    print("=" * 60)
    print("Stopping SportsBot...")
    print("=" * 60)

    await close_mongo_connection()


app = FastAPI(
    title="SportsBot API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

app.include_router(chatbot_router)
from app.routes.ingest import router as ingest_router
app.include_router(ingest_router)


@app.get("/")
def home():

    return {
        "message": "Vanakkam Da Mapla Nan Dhan Sports Bot Backend Server"
    }


@app.get("/health")
async def health():

    return {"status": "OK", "mongo_connected": db_instance.client is not None, "vector_count": await get_vector_collection().estimated_document_count()}