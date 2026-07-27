import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from app.routes.ingest import ingest_now
from app.db.mongo_db import connect_to_mongo, close_mongo_connection

async def test_ingest():
    await connect_to_mongo()
    print(await ingest_now())
    await close_mongo_connection()

asyncio.run(test_ingest())
