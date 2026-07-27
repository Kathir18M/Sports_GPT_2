import sys
import os

# Add bot_backend to path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
import logging

# Setup Logger (Critical for Cloud Debugging)
logger = logging.getLogger("uvicorn")

class Database:
    client: AsyncIOMotorClient = None

db_instance = Database()

def get_database_client():
    return db_instance.client


def get_vector_collection():
        db_name = getattr(settings, "MONGO_DB_NAME", None) or "sports_rag"
        return db_instance.client[db_name]["vector_documents"]


async def connect_to_mongo():
    try:
        logger.info("⏳ Connecting to MongoDB...")
        # Use TLS option that relaxes certificate verification for local development
        db_instance.client = AsyncIOMotorClient(settings.MONGO_URL, tlsAllowInvalidCertificates=True)
        
        # THE PING TEST (Crucial for Cloud)
        await db_instance.client.admin.command('ping')
        logger.info("✅ MongoDB Connected Successfully!")
    except Exception as e:
        logger.error(f"❌ MongoDB Connection Failed: {e}")
        raise e


async def close_mongo_connection():
    if db_instance.client:
        db_instance.client.close()
        logger.info("🔒 MongoDB connection closed.")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        asyncio.run(connect_to_mongo())
    finally:
        asyncio.run(close_mongo_connection())