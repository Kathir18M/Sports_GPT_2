import asyncio
from app.db.mongo_db import connect_to_mongo, close_mongo_connection

async def main():
    try:
        await connect_to_mongo()
        print("✅ MongoDB connection succeeded")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)
    finally:
        await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(main())
