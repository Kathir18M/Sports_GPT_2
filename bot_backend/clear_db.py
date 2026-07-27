import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from app.db.mongo_db import connect_to_mongo, close_mongo_connection, get_vector_collection

async def clear():
    await connect_to_mongo()
    col = get_vector_collection()
    await col.delete_many({})
    print('Cleared vector_documents')
    await close_mongo_connection()

asyncio.run(clear())
